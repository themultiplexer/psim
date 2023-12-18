#version 330 core

in vec3 position;
in vec3 bary;

out vec3 b;
out vec3 p;

void main() {
    gl_Position = vec4(position, 1.0);
    p = position;
    b = bary;
}