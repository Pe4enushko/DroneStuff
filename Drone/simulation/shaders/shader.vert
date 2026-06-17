#version 330

in vec3 in_position;
in vec3 in_color;

out vec3 v_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(in_position, 1.0);
    v_color = in_color;
}