#version 330 core

uniform sampler2D texture;
in vec2 coords;

void main(){
    gl_FragColor = texture2D(texture, coords);
}
