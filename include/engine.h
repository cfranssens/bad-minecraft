#ifndef ENGINE_H
#define ENGINE_H

#include <unordered_map>
#include <vector>



// All resource types
enum GLType {
    VAO,
    Buffer,
    Shader, 
    ShaderProgram,
    Texture,
};

// Represents a resource in gpu memory, anything from a VBO, VAO, Buffer
struct Resource {
    GLuint binding;
    GLType ty;
};


// Resources class

// This structure holds the pointers to all OpenGL resources in a neat way. 
class Resources {
    public:
        Resources();
        ~Resources();

        // Attach resources to the unordered_map
        unsigned int attach_buffer(const void* contents, size_t size, GLenum binding_target, GLenum draw_mode);
        unsigned int attach_shader(const char* source, GLenum ty);
        unsigned int attach_texture(const char* source, GLenum ty);

        void attach_shader(unsigned int index, const char* source, GLenum ty);
        
        unsigned int attach_shader_program(unsigned int n, unsigned int* shaders);
        void attach_shader_program(unsigned int index, unsigned int n, unsigned int* shaders);

   
        unsigned int attach_vao();
        void vertex_attribute(GLbyte index, unsigned int size, GLenum type, bool normalized, unsigned int stride, void* pointer);

        // Use and bind resources
        void use_shader_program(unsigned int index);
        void bind_vao(unsigned int index);
        void bind_buffer(unsigned int index, GLenum ty);
        void bind_buffer_base(unsigned int index, unsigned int shader_index, GLenum ty);  

        // Shader operations
        GLuint get_uniform_location(unsigned int index, char* uniform); 

        // Retrieve an OpenGL binding in resource 
        Resource get(unsigned int index);
        unsigned int get_first_empty();
        void remove(unsigned int);


    private:
        // The bindings to the resources
        std::unordered_map<unsigned int, Resource> resources;
};

#endif 
