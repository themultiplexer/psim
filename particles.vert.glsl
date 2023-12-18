#version 330 core

in vec2 position;
in vec4 color;
in float pointsize;
in float mass;

uniform vec2 iResolution;
uniform float zoom;
uniform vec2 offset;

out vec4 c;

void main(){
    vec2 newpos = position + offset * vec2(3.5555f,2.0f) / zoom;
    float AR = iResolution.y / iResolution.x;
    newpos.x *= AR;
    newpos *= zoom;

    c = color;
    gl_Position = vec4(newpos, mass, 1.0);
    gl_PointSize = pointsize * clamp(zoom, 0.1f, 0.9f) + 0.5f;
}
