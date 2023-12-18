#version 330 core

in vec2 position;

out vec2 coords;

void main(){
    gl_Position = vec4(position, 0.0, 1.0);
    coords = (position + 1.0f) / 2.0f;
}
