#include <glad/glad.h> 
#include <engine.h>
#include <iostream>

#include <string>
#include <fstream>
#include <sstream>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Delete a resource
void del(Resource res) {
    std::cout << res.ty << ", " << res.binding << std::endl;
    switch (res.ty) {
        case GLType::VAO:
            glDeleteVertexArrays(1, &res.binding);
            break;
        case GLType::Buffer:
            glDeleteBuffers(1, &res.binding);
            break;
        case GLType::Shader:
            glDeleteShader(res.binding);
            break;
        case GLType::ShaderProgram:
            glDeleteProgram(res.binding);
            break;
        case GLType::Texture:
            glDeleteTextures(1, &res.binding);
            break;
    }
}

// Unused constructor
Resources::Resources() {
}

// Delete all GPU resources
Resources::~Resources() { 
    for (const auto& pair : resources) {
        unsigned int key = pair.first;
        Resource res = pair.second;
        del(res);
    }
}

// Get the first empty map in the unordered map
unsigned int Resources::get_first_empty() {
    unsigned int index = 0;
    while (resources.find(index) != resources.end()) {
        index ++;
    }

    return index;
}


Resource Resources::get(unsigned int index) {
    return resources[index];
}

void Resources::remove(unsigned int index) {
    del(resources[index]);
    resources.erase(index);
}

unsigned int Resources::attach_buffer(const void* contents, size_t size, GLenum binding_target, GLenum draw_mode) {
    GLuint buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(binding_target, buffer);
    glBufferData(binding_target, size, contents, draw_mode);
    glBindBuffer(binding_target, 0);
    
    Resource resource;
    resource.binding = buffer;
    resource.ty = GLType::Buffer;

    unsigned int index = this->get_first_empty();
    resources[index] = resource;
    return index;
}


unsigned int Resources::attach_texture(const char* fp, GLenum ty) {   
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(ty, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    int width, height, nrChannels;
    unsigned char *data = stbi_load(fp, &width, &height, &nrChannels, 0);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);
    return texture;
}

// Attaches a combination of shaders i.e a pipeline
unsigned int Resources::attach_shader_program(unsigned int n, unsigned int* shaders) {
    int  success;
    char info_log[512];

    unsigned int shader_program;
    shader_program = glCreateProgram();

    for (unsigned int shader = 0; shader < n; shader ++) {
        glAttachShader(shader_program, resources[shaders[shader]].binding);
    }

    glLinkProgram(shader_program);
    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(shader_program, 512, nullptr, info_log);
        std::cout << "ERROR::SHADER_PROGRAM::" << shaders << "::LINKING_FAILED\n" << info_log << std::endl;
    }

    Resource resource;
    resource.binding = shader_program;
    resource.ty = GLType::ShaderProgram;
    
    unsigned int index = this->get_first_empty();
    resources[index] = resource;
    return index;
}


// Attach a shader program to a specific mapping
void Resources::attach_shader_program(unsigned int index, unsigned int n, unsigned int* shaders) {
    int  success;
    char info_log[512];

    del(resources[index]);


    unsigned int shader_program;
    shader_program = glCreateProgram();

    for (unsigned int shader = 0; shader < n; shader ++) {
        glAttachShader(shader_program, resources[shaders[shader]].binding);
    }

    glLinkProgram(shader_program);
    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(shader_program, 512, nullptr, info_log);
        std::cout << "ERROR::SHADER_PROGRAM::" << shaders << "::LINKING_FAILED\n" << info_log << std::endl;
    }

    Resource resource;
    resource.binding = shader_program;
    resource.ty = GLType::ShaderProgram;
    
    resources[index] = resource;
}


// Load a shader from disk
unsigned int Resources::attach_shader(const char* fp, GLenum ty) {
    std::string shader_code;
    std::ifstream shader_file;
    
    shader_file.exceptions (std::ifstream::failbit | std::ifstream::badbit);
    
    try {
        shader_file.open(fp);
        std::stringstream shader_stream;
        shader_stream << shader_file.rdbuf();		
        shader_file.close();
        shader_code = shader_stream.str();
    } catch (std::ifstream::failure& e) {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
    }
        
    const char* source = shader_code.c_str();
    //std::cout << source << std::endl;

    int success;
    char info_log[512];

    unsigned int shader;
    shader = glCreateShader(ty);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(shader, 512, NULL, info_log);
        std::cout << "ERROR::SHADER::" << fp << "::" << ty << "::COMPILATION_FAILED\n" << info_log << std::endl;
    }

    Resource resource;
    resource.binding = shader;
    resource.ty = GLType::Shader;

    unsigned int index = this->get_first_empty();
    resources[index] = resource;
    return index;
}

// Attach shader to specific mapping
void Resources::attach_shader(unsigned int index, const char* fp, GLenum ty) {
    std::string shader_code;
    std::ifstream shader_file;
    
    shader_file.exceptions (std::ifstream::failbit | std::ifstream::badbit);
    
    try {
        shader_file.open(fp);
        std::stringstream shader_stream;
        shader_stream << shader_file.rdbuf();		
        shader_file.close();
        shader_code = shader_stream.str();
    } catch (std::ifstream::failure& e) {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
    }
        
    const char* source = shader_code.c_str();
    del(resources[index]);


    int success;
    char info_log[512];

    unsigned int shader;
    shader = glCreateShader(ty);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(shader, 512, NULL, info_log);
        std::cout << "ERROR::SHADER::" << fp << "::" << ty << "::COMPILATION_FAILED\n" << info_log << std::endl;
    }

    Resource resource;
    resource.binding = shader;
    resource.ty = GLType::Shader;

    resources[index] = resource;
}

unsigned int Resources::attach_vao() {
    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
   
    Resource resource;
    resource.binding = VAO;
    resource.ty = GLType::VAO;

    unsigned int index = this->get_first_empty();
    resources[index] = resource;
    return index;
}

// Set VAO attributes
void Resources::vertex_attribute(GLbyte index, unsigned int size, GLenum type, bool normalized, unsigned int stride, void* pointer) {
    glVertexAttribPointer(index, size, type, normalized, stride, pointer);
    glEnableVertexAttribArray(index);
}


// Bind objects

void Resources::bind_vao(unsigned int index) {
    Resource res = resources[index];
    if (res.ty == GLType::VAO && resources.find(index) != resources.end()) {
        glBindVertexArray(res.binding);
    } else {
        std::cout << "resources[" << index << "] does not contain GLType::VAO" << std::endl;
    }
}

void Resources::use_shader_program(unsigned int index) {
    Resource res = resources[index];
    if (res.ty == GLType::ShaderProgram && resources.find(index) != resources.end()) {
        glUseProgram(res.binding);
    } else {
        std::cout << "resources[" << index << "] does not contain GLType::ShaderProgram" << std::endl;
    }
}

void Resources::bind_buffer(unsigned int index, GLenum ty) {
    Resource res = resources[index];
    if (res.ty == GLType::Buffer && resources.find(index) != resources.end()) {
        glBindBuffer(ty, res.binding);
    } else {
        std::cout << "resources[" << index << "] does not contain GLType::Buffer (" << res.ty << ")" << std::endl;
    }
}

void Resources::bind_buffer_base(unsigned int index, unsigned int shader_index, GLenum ty) {
    Resource res = resources[index];
    if (res.ty == GLType::Buffer && resources.find(index) != resources.end()) {
        glBindBufferBase(ty, shader_index, res.binding);
    } else {
        std::cout << "resources[" << index << "] does not contain GLType::Buffer" << std::endl;
    }
}





// Shader interaction
GLuint Resources::get_uniform_location(unsigned int index, char* uniform) {
    Resource res = resources[index];
    if (res.ty == GLType::ShaderProgram && resources.find(index) != resources.end()) {
        glUseProgram(res.binding);
    } else {
        std::cout << "resources[" << index << "] does not contain GLType::ShaderProgram" << std::endl;
        return 0;
    }

    return glGetUniformLocation(res.binding, uniform);
}



