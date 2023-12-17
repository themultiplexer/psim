in vec2 position;
in vec4 color;
in float pointsize;

uniform vec2 iResolution;
uniform float zoom;

void main(){

    float AR = iResolution.y / iResolution.x;
    position.x *= AR;

    position *= zoom;

    gl_FrontColor = color;
    gl_Position = vec4(position, 1.0, 1.0);
    gl_PointSize = pointsize * clamp(zoom, 0.1f, 0.9f) + 0.2f;
}
