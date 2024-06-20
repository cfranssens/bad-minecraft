#version 450 core
layout (local_size_x = 32, local_size_y = 6, local_size_z = 1) in;


// Noise functions
// https://iquilezles.org/articles/morenoise/

float hash( ivec3 p )  {                        // replace this by something better

    // 3D -> 1D
    int n = p.x*3 + p.y*113 + p.z*311;

    // 1D hash by Hugo Elias
	n = (n << 13) ^ n;
    n = n * (n * n * 15731 + 789221) + 1376312589;
    return -1.0+2.0*float( n & 0x0fffffff)/float(0x0fffffff);
}

// 2D noise function with gradient computation
vec4 noised(in vec3 x) {
    ivec3 p = ivec3(floor(x));        // Integer part of x
    vec3 w = fract(x);        // Fractional part of x

    vec3 u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);  // Smooth interpolation
    vec3 du = 30.0 * w * w * (w * (w - 2.0) + 1.0);      // Derivative of u

    // Random values at the corners of the unit cube
    float a = hash(p + ivec3(0, 0, 0));
    float b = hash(p + ivec3(1, 0, 0));
    float c = hash(p + ivec3(0, 1, 0));
    float d = hash(p + ivec3(1, 1, 0));
    float e = hash(p + ivec3(0, 0, 1));
    float f = hash(p + ivec3(1, 0, 1));
    float g = hash(p + ivec3(0, 1, 1));
    float h = hash(p + ivec3(1, 1, 1));

    // Interpolation coefficients
    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k3 = e - a;
    float k4 = a - b - c + d;
    float k5 = a - c - e + g;
    float k6 = a - b - e + f;
    float k7 = -a + b + c - d + e - f - g + h;

    // Compute the noise value
    return vec4( k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z, 
                 du * vec3( k1 + k4*u.y + k6*u.z + k7*u.y*u.z,
                            k2 + k5*u.z + k4*u.x + k7*u.z*u.x,
                            k3 + k6*u.x + k5*u.y + k7*u.x*u.y ) );
}

vec4 fbm( in vec3 x, int octaves )
{
        const float scale  = 1.5;

    float a = 0.0;
    float b = 0.5;
	float f = 1.0;
    vec3  d = vec3(0.0);
    for( int i=0; i<octaves; i++ )
    {
        vec4 n = noised(f*x*scale);
        a += b*n.x;           // accumulate values		
        d += b*n.yzw*f*scale; // accumulate derivatives
        b *= 0.5;             // amplitude decrease
        f *= 1.8;             // frequency increase
    }

	return vec4( a, d );
}


struct Vertex {
    vec3 position;
    float _pad;
    vec2 uv;
    uint axis;
    uint block_hash;
};

// Currently store vertices/indices directly in these, working on handling it in memory pools
layout(std430, binding = 0) buffer vertex_buffer {
    Vertex debug_vertices[];
};

layout(std430, binding = 1) buffer index_buffer {
    uint debug_indices[];
};

const uint VOXEL_TYPES = 32; 

struct Chunk {
    ivec3 position; // Position in chunk space
    int block_data[32 * 32 * 32]; // This currently describes the block at the position within the chunk
    uint axis_columns[6 * 32 * 32]; 
    uint data[6][VOXEL_TYPES][32][32]; // Binary masks
    
    // Pooling related vars
    uint n;
    uint pool_references[256];
    uint bit_table[8]; 
};


layout(std430, binding = 2) buffer chunk_data_buffer {
    Chunk chunk_data[];
};


// Command struct
struct Command {
    ivec3 position; // Position of the chunk
    int state; // Whether it should be remeshed or generated
};

layout(std430, binding = 3) buffer command_buffer {
    Command commands[];
};

// Work in progress
layout (binding = 4, offset = 0) uniform atomic_uint fm_pool;
layout (binding = 4, offset = 4) uniform atomic_uint qo;



struct Pool {
    uint chunk_index; // Index to the active chunk in memory, relative
    uint n; // how much of it is filled
    uint quads[256]; // packed data for quads, in chunk local space
};

layout(std430, binding = 5) buffer memory_pool_buffer {
    Pool memory_pools[];
};

layout(std430, binding = 6) buffer memory_pool_desc_buffer {
    uint table[4096];
};


uniform uint CHUNK_ID_BUFFER_SIZE;
uniform uint MEMORY_POOL_SIZE;
uniform uint N_MEMORY_POOLS;


// Return number of unsignificant bits
uint trailing_zeroes(uint x) {
    if (x == 0u) return 32; // If the input is 0, all bits are zeros

    int count = 0;
    while ((x & 1u) == 0u) {
        x >>= 1u;
        count++;
    }
    return count;
}

uint trailing_ones(uint x) {
    if (x == 0u) return 32; // If the input is 0, all bits are zeros

    int count = 0;
    while ((x & 1u) == 1u) {
        x >>= 1u;
        count++;
    }
    return count;
}


// Separate functions because GLSL doesnt allow passing dynamic sized arrays
// Search the bit table for the first empty pool
uint search_bit_table_4096(uint table[4096]) {
    for (uint j = 0; j < 4096; ++j) { // if you have more than 128 chunks in memory, damn you have a good computer but also this code will fail. 
        uint value = table[j];
        for (uint i = 0; i < 32; ++i) {
            if ((value & (1u << i)) == 0) {
                return j * 32u + i;
            }
        }
    }
    return 4096u * 32u;
}

// Search the bit table for the first empty pool
uint search_bit_table_8(uint table[8]) {
    for (uint j = 0; j < 8; ++j) {         
        uint value = table[j];
        for (uint i = 0; i < 32; ++i) {
            if ((value & (1u << i)) == 0) {
                return j * 32u + i;
            }
        }
    }
    return 8u * 32u;
}

// Flip specific bit
void flip_bit_table_8(inout uint table[8], uint i) {
    if (i < 128u) {
        const uint BITS_PER_UINT = 32u;

        uint e = i / BITS_PER_UINT;
        uint b = i & (BITS_PER_UINT - 1);
    
        table[e] ^= (1u << b);
    }
}


// Replace this with something better
void add_debug_quad(float row, float y, float w, float h, uint axis, uint block_hash, ivec3 chunk_pos, uint axis_dist, uint index) {
   uint chunk_size = 30;
   float scale = 150.0;

        if (axis == 0) {
            vec3 offset = vec3(float(y) / scale + float(chunk_pos.x * chunk_size) / scale, float(axis_dist) / scale + float(chunk_pos.y * chunk_size) / scale, float(row) / scale + float(chunk_pos.z * chunk_size) / scale);
            debug_vertices[index * 4    ].position = offset;
            debug_vertices[index * 4    ].uv = vec2(0.0, 0.0);
            debug_vertices[index * 4    ].axis = axis;
            debug_vertices[index * 4    ].block_hash = block_hash;


            debug_vertices[index * 4 + 1].position = offset + vec3(float(h) / scale, 0.0, 0.0);
            debug_vertices[index * 4 + 1].uv = vec2(float(h), 0.0);
            debug_vertices[index * 4 + 1].axis = axis;
            debug_vertices[index * 4 + 1].block_hash = block_hash;


            debug_vertices[index * 4 + 2].position = offset + vec3(0.0, 0.0, float(w) / scale);
            debug_vertices[index * 4 + 2].uv = vec2(0.0, float(w));
            debug_vertices[index * 4 + 2].axis = axis;
            debug_vertices[index * 4 + 2].block_hash = block_hash;

            debug_vertices[index * 4 + 3].position = offset + vec3(float(h) / scale, 0.0, float(w) / scale);
            debug_vertices[index * 4 + 3].uv = vec2(float(h), float(w));
            debug_vertices[index * 4 + 3].axis = axis;
            debug_vertices[index * 4 + 3].block_hash = block_hash;

            debug_indices[index * 6] = index * 4;
            debug_indices[index * 6 + 1] = index * 4 + 1;
            debug_indices[index * 6 + 2] = index * 4 + 2;

            debug_indices[index * 6 + 3] = index * 4 + 1;
            debug_indices[index * 6 + 4] = index * 4 + 3;
            debug_indices[index * 6 + 5] = index * 4 + 2;
        }

        if (axis == 1) {
            vec3 offset = vec3(float(y) / scale + float(chunk_pos.x * chunk_size) / scale, float(axis_dist + 1) / scale + float(chunk_pos.y * chunk_size) / scale, float(row) / scale + float(chunk_pos.z * chunk_size) / scale);
            debug_vertices[index * 4    ].position = offset;
            debug_vertices[index * 4    ].uv = vec2(0.0, 0.0);
            debug_vertices[index * 4    ].axis = axis;
            debug_vertices[index * 4    ].block_hash = block_hash;

            debug_vertices[index * 4 + 1].position = offset + vec3(float(h) / scale, 0.0, 0.0);
            debug_vertices[index * 4 + 1].uv = vec2(float(h), 0.0);
            debug_vertices[index * 4 + 1].axis = axis;
            debug_vertices[index * 4 + 1].block_hash = block_hash;

            debug_vertices[index * 4 + 2].position = offset + vec3(0.0, 0.0, float(w) / scale);
            debug_vertices[index * 4 + 2].uv = vec2(0.0, float(w));                    
            debug_vertices[index * 4 + 2].axis = axis;
            debug_vertices[index * 4 + 2].block_hash = block_hash;
            
            debug_vertices[index * 4 + 3].position = offset + vec3(float(h) / scale, 0.0, float(w) / scale);
            debug_vertices[index * 4 + 3].uv = vec2(float(h), float(w));
            debug_vertices[index * 4 + 3].axis = axis;
            debug_vertices[index * 4 + 3].block_hash = block_hash;

            debug_indices[index * 6] = index * 4;
            debug_indices[index * 6 + 2] = index * 4 + 1;
            debug_indices[index * 6 + 1] = index * 4 + 2;

            debug_indices[index * 6 + 3] = index * 4 + 1;
            debug_indices[index * 6 + 5] = index * 4 + 3;
            debug_indices[index * 6 + 4] = index * 4 + 2;
        }



        if (axis == 2) {
            vec3 offset = vec3(float(row) / scale + float(chunk_pos.x * chunk_size) / scale, float(y) / scale + float(chunk_pos.y * chunk_size) / scale, float(axis_dist) / scale + float(chunk_pos.z * chunk_size) / scale);

            debug_vertices[index * 4    ].position = offset;
            debug_vertices[index * 4    ].uv = vec2(0.0, 0.0);
            debug_vertices[index * 4    ].axis = axis;
            debug_vertices[index * 4    ].block_hash = block_hash;


            debug_vertices[index * 4 + 1].position = offset + vec3(float(w) / scale, 0.0, 0.0);
            debug_vertices[index * 4 + 1].uv = vec2(float(w), 0.0);
            debug_vertices[index * 4 + 1].axis = axis;
            debug_vertices[index * 4 + 1].block_hash = block_hash;

            debug_vertices[index * 4 + 2].position = offset + vec3(0.0, float(h) / scale, 0.0);
            debug_vertices[index * 4 + 2].uv = vec2(0.0, float(h));
            debug_vertices[index * 4 + 2].axis = axis;
            debug_vertices[index * 4 + 2].block_hash = block_hash;

            debug_vertices[index * 4 + 3].position = offset + vec3(float(w) / scale, float(h) / scale, 0.0);
            debug_vertices[index * 4 + 3].uv = vec2(float(w), float(h));
            debug_vertices[index * 4 + 3].axis = axis;
            debug_vertices[index * 4 + 3].block_hash = block_hash;


            debug_indices[index * 6] = index * 4;
            debug_indices[index * 6 + 2] = index * 4 + 1;
            debug_indices[index * 6 + 1] = index * 4 + 2;

            debug_indices[index * 6 + 3] = index * 4 + 1;
            debug_indices[index * 6 + 5] = index * 4 + 3;
            debug_indices[index * 6 + 4] = index * 4 + 2;
        }

        if (axis == 3) {
            vec3 offset = vec3(float(row) / scale + float(chunk_pos.x * chunk_size) / scale, float(y) / scale + float(chunk_pos.y * chunk_size) / scale, float(axis_dist + 1) / scale + float(chunk_pos.z * chunk_size) / scale);
            debug_vertices[index * 4    ].position = offset;
            debug_vertices[index * 4    ].uv = vec2(0.0, 0.0);
            debug_vertices[index * 4    ].axis = axis;
            debug_vertices[index * 4    ].block_hash = block_hash;


            debug_vertices[index * 4 + 1].position = offset + vec3(float(w) / scale, 0.0, 0.0);
            debug_vertices[index * 4 + 1].uv = vec2(float(w), 0.0);
            debug_vertices[index * 4 + 1].axis = axis;
            debug_vertices[index * 4 + 1].block_hash = block_hash;

            debug_vertices[index * 4 + 2].position = offset + vec3(0.0, float(h) / scale, 0.0);
            debug_vertices[index * 4 + 2].uv = vec2(0.0, float(h));
            debug_vertices[index * 4 + 2].axis = axis;
            debug_vertices[index * 4 + 2].block_hash = block_hash;

            debug_vertices[index * 4 + 3].position = offset + vec3(float(w) / scale, float(h) / scale, 0.0);
            debug_vertices[index * 4 + 3].uv = vec2(float(w), float(h));
            debug_vertices[index * 4 + 3].axis = axis;
            debug_vertices[index * 4 + 3].block_hash = block_hash;


            debug_indices[index * 6] = index * 4;
            debug_indices[index * 6 + 1] = index * 4 + 1;
            debug_indices[index * 6 + 2] = index * 4 + 2;

            debug_indices[index * 6 + 3] = index * 4 + 1;
            debug_indices[index * 6 + 4] = index * 4 + 3;
            debug_indices[index * 6 + 5] = index * 4 + 2;      

        }
        if (axis == 4) {
            vec3 offset = vec3(float(axis_dist) / scale + float(chunk_pos.x * chunk_size) / scale, float(y) / scale + float(chunk_pos.y * chunk_size) / scale, float(row) / scale + float(chunk_pos.z * chunk_size) / scale);
            debug_vertices[index * 4    ].position = offset;
            debug_vertices[index * 4    ].uv = vec2(0.0, 0.0);
            debug_vertices[index * 4    ].axis = axis;
            debug_vertices[index * 4    ].block_hash = block_hash;


            debug_vertices[index * 4 + 1].position = offset + vec3(0.0, float(h) / scale, 0.0);
            debug_vertices[index * 4 + 1].uv = vec2(float(h), 0.0);
            debug_vertices[index * 4 + 1].axis = axis;
            debug_vertices[index * 4 + 1].block_hash = block_hash;

         
            debug_vertices[index * 4 + 2].position = offset + vec3(0.0, 0.0, float(w) / scale);
            debug_vertices[index * 4 + 2].uv = vec2(0.0, float(w));
            debug_vertices[index * 4 + 2].axis = axis;
            debug_vertices[index * 4 + 2].block_hash = block_hash;

          
            debug_vertices[index * 4 + 3].position = offset + vec3(0.0, float(h) / scale, float(w) / scale);
            debug_vertices[index * 4 + 3].uv = vec2(float(h), float(w));
            debug_vertices[index * 4 + 3].axis = axis;
            debug_vertices[index * 4 + 3].block_hash = block_hash;



            debug_indices[index * 6] = index * 4;
            debug_indices[index * 6 + 2] = index * 4 + 1;
            debug_indices[index * 6 + 1] = index * 4 + 2;

            debug_indices[index * 6 + 3] = index * 4 + 1;
            debug_indices[index * 6 + 5] = index * 4 + 3;
            debug_indices[index * 6 + 4] = index * 4 + 2;
        }
        if (axis == 5) {
            vec3 offset = vec3(float(axis_dist + 1) / scale + float(chunk_pos.x * chunk_size) / scale, float(y) / scale + float(chunk_pos.y * chunk_size) / scale, float(row) / scale + float(chunk_pos.z * chunk_size) / scale);
            debug_vertices[index * 4    ].position = offset;
            debug_vertices[index * 4    ].uv = vec2(0.0, 0.0);
            debug_vertices[index * 4    ].axis = axis;
            debug_vertices[index * 4    ].block_hash = block_hash;


            debug_vertices[index * 4 + 1].position = offset + vec3(0.0, float(h) / scale, 0.0);
            debug_vertices[index * 4 + 1].uv = vec2(float(h), 0.0);
            debug_vertices[index * 4 + 1].axis = axis;
            debug_vertices[index * 4 + 1].block_hash = block_hash;

         
            debug_vertices[index * 4 + 2].position = offset + vec3(0.0, 0.0, float(w) / scale);
            debug_vertices[index * 4 + 2].uv = vec2(0.0, float(w));
            debug_vertices[index * 4 + 2].axis = axis;
            debug_vertices[index * 4 + 2].block_hash = block_hash;

          
            debug_vertices[index * 4 + 3].position = offset + vec3(0.0, float(h) / scale, float(w) / scale);
            debug_vertices[index * 4 + 3].uv = vec2(float(h), float(w));
            debug_vertices[index * 4 + 3].axis = axis;
            debug_vertices[index * 4 + 3].block_hash = block_hash;



            debug_indices[index * 6] = index * 4;
            debug_indices[index * 6 + 1] = index * 4 + 1;
            debug_indices[index * 6 + 2] = index * 4 + 2;

            debug_indices[index * 6 + 3] = index * 4 + 1;
            debug_indices[index * 6 + 4] = index * 4 + 3;
            debug_indices[index * 6 + 5] = index * 4 + 2;
        }
    }


// The actual greedy meshing code
// This takes in a 32x32 grid of binary data, what block type it is, the index into the chunk, the axis, and distance down the axis.

void mesh_bin_plane(uint plane[32], uint block_hash, ivec3 chunk_pos, uint chunk_index, uint axis, uint axis_dist) {
    Chunk chunk = chunk_data[chunk_index];

    // Iterate over all rows

    for (uint row = 0; row < 32; row ++) {
        uint y = 0;

        while (y < 32) {
            y += trailing_zeroes(plane[row] >> y); // Count number of 0s
            if (y >= 32) {continue;} // If it is more than there are spaces in the plane, continue with the next iteration of 'row'
                
            uint h = trailing_ones(plane[row] >> y); // Get height
            uint h_mask = (1u << h) - 1u;            // Binary representation of h 
            uint mask = h_mask << y;                 // Shift to y


            uint w = 1; // Width 
            while ((row + w) < 32) { 
                uint next_row_h = (plane[row + w] >> y) & h_mask; // Get next row as binary, with a length of h
                if (next_row_h != h_mask) {                       // If it isn't the same, stop extending the quad 
                    break;
                } 
                plane[row + w] = plane[row + w] & ~mask;          // Remove the next row's mask to avoid meshing already meshed sections
                w += 1;
            }

            barrier();
        
            
            // Memory pooling
            uint n = chunk_data[chunk_index].n;
            uint m_pool = chunk_data[chunk_index].pool_references[n];
            memory_pools[m_pool].n += 1; // Add quad here

            barrier();
            
            // Add the quad to a temporary vertex buffer
            uint x = atomicCounter(fm_pool);                             // (Only 20 block colors available at the moment)
            add_debug_quad(float(row), float(y), float(w), float(h), axis, block_hash, chunk_pos, axis_dist, atomicCounterIncrement(qo));
            memoryBarrierShared();
            barrier();

            // More memory pooling
            if (memory_pools[m_pool].n >= 256) {
                flip_bit_table_8(chunk_data[chunk_index].bit_table, n); // set table's state to filled
                chunk_data[chunk_index].n = search_bit_table_8(chunk_data[chunk_index].bit_table); // get first empty table in the chunk
                atomicCounterIncrement(fm_pool);
                memoryBarrierShared();
                barrier();
            }

            y += h;                
        }
    }


}


void main() {
    // The chunk index, each cinvocation handles one chunk, and should only be dispatched in one dimension
    uint chunk_index = gl_WorkGroupID.x; 
    ivec3 chunk_position = commands[chunk_index].position;

    // This is not 32 because of the padding at the beginning and end to account for face culling in between chunks
    uint chunk_size = 30;
 
    // Generate chunk data, only when the command is 0
    if (commands[chunk_index].state == 0) {
        uint i = gl_LocalInvocationID.x;    // range from 0 to 31
        uint axis = gl_LocalInvocationID.y; // range from 0 to 5
        
        // Clear everything in the chunk
        chunk_data[chunk_index].n = 0;
        for (uint i = 0; i < 128u; i ++) {
            chunk_data[chunk_index].pool_references[i] = 0;
            memory_pools[i].n = 0;
        }

        for (uint y = 0; y < 32; y ++) {
            for (uint z = 0; z < 32; z ++) {
                chunk_data[chunk_index].axis_columns[y + (z * 32)] = 0u;
                chunk_data[chunk_index].axis_columns[y + (z * 32) + 1024] = 0u;
                chunk_data[chunk_index].axis_columns[y + (z * 32) + 1024 * 2] = 0u;
                
                for (uint block = 0; block < VOXEL_TYPES; block ++) {
                    chunk_data[chunk_index].data[axis][block][y][z] = 0;
                }
            }
        }

        barrier(); // Sync threads to ensure eveything is cleared

        // Generate some noise (just for testing)
        for (uint y = 0; y < 32; y ++) {
            for (uint z = 0; z < 32; z ++) {
                uint x = i;
                uint index = (z * 32 * 32) + (y * 32) + x;

                vec3 world_pos = vec3(float(x) + float(chunk_position.z * chunk_size),
                                      float(y) + float(chunk_position.y * chunk_size),
                                      float(z) + float(chunk_position.x * chunk_size)) / 50.0;
                vec4 noise = fbm(world_pos, 3); // Assuming fbm function returns a vec4 with noise components
                int t_height = int(round(noise.x * 16.0));

                int block_type = 0; // Default block type (void)

                // Used for testing different block types
                if (noise.x > 0.2) {
                    block_type = 1;
                } else if (noise.x > 0.15) {
                    block_type = 2;
                } else if (noise.x > 0.1) {
                    block_type = 3;
                } else if (noise.x > 0.05) {
                    block_type = 4;
                } else if (noise.x > 0.04) {
                    block_type = 5;
                } else if (noise.x > 0.03) {
                    block_type = 6;
                } else if (noise.x > 0.02) {
                    block_type = 7;
                } else if (noise.x > 0.01) {
                    block_type = 8;
                } else if (noise.x > 0.005) {
                    block_type = 9;
                } else if (noise.x > 0.001) {
                    block_type = 10;
                } else if (t_height > 10) {
                    block_type = 11;
                } else if (t_height > 9) {
                    block_type = 12;
                } else if (t_height > 8) {
                    block_type = 13;
                } else if (t_height > 7) {
                    block_type = 14;
                } else if (t_height > 6) {
                    block_type = 15;
                } else if (t_height > 5) {
                    block_type = 16;
                } else if (t_height > 4) {
                    block_type = 17;
                } else if (t_height > 3) {
                    block_type = 18;
                } else if (t_height > 2) {
                    block_type = 19;
                } else if (t_height > 1) {
                    block_type = 20;
                } else {
                    block_type = 0; // Default block type
                }

                chunk_data[chunk_index].block_data[index] = block_type;
            }
        }
    
        barrier();

        for (uint y = 0; y < 32; y ++) {
            for (uint z = 0; z < 32; z ++) {
                for (uint x = 0; x < 32; x ++) {
                    uint index = (z * 32u * 32u) + (y * 32u) + x; // Loop 0..CHUNK_SIZE_P2

                    if (chunk_data[chunk_index].block_data[index] != 0) {
                        chunk_data[chunk_index].axis_columns[x + (z * 32)] |= 1u << y;
                        chunk_data[chunk_index].axis_columns[z + (y * 32) + 1024] |= 1u << x;
                        chunk_data[chunk_index].axis_columns[x + (y * 32) + 1024 * 2] |= 1u << z;
                    } 
                }
            }
        }

        barrier();


        if (axis < 3) {
            for (uint j = 0; j < 32; j ++) {
                uint t = i * 32 + j;
                uint col = chunk_data[chunk_index].axis_columns[(1024 * axis) + t];
                chunk_data[chunk_index].axis_columns[(1024 * (axis * 2 + 1)) + t] = col & ~(col >> 1u);
                chunk_data[chunk_index].axis_columns[(1024 * (axis * 2 + 0)) + t] = col & ~(col << 1u);
            }
        }

        //barrier();

        bool block_hashes[VOXEL_TYPES];
        uint g = 0;

        for (uint z = 0; z < 30u; z ++) {
            for (uint x = 0; x < 30u; x ++) {
                uint col_index = 1u + x + ((z + 1u) * 32u) + 1024u * axis;
                uint col = chunk_data[chunk_index].axis_columns[col_index] >> 1u;
                col &= ~(1u << 30u);
                       
                while (col != 0u) {
                    uint y = trailing_zeroes(col);
                    col &= col - 1u;

                    uvec3 voxel_pos = uvec3(x, z, y);
                    if (axis == 0 || axis == 1) {voxel_pos = uvec3(x, y, z);}
                    if (axis == 2 || axis == 3) {voxel_pos = uvec3(y, z, x);}

                    int block_hash = chunk_data[chunk_index].block_data[(voxel_pos.z + 1) * 1024 + (voxel_pos.y + 1) * 32 + voxel_pos.x + 1]; 
                    if (block_hash != 0) {
                        chunk_data[chunk_index].data[axis][block_hash][y][x] |= 1u << z;
                        block_hashes[block_hash] = true;
                    }
                } 
            }
        }

        barrier();

        for (uint block = 0; block < VOXEL_TYPES; block ++) {
            if (block_hashes[block]) {
                mesh_bin_plane(chunk_data[chunk_index].data[axis][block][i], block, chunk_position, chunk_index, axis, i);
            }
        }
    }
}

 
