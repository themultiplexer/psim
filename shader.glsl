in vec2 position;
in vec4 color;
in float pointsize;

void main(){
    gl_FrontColor = color;
    gl_Position = vec4(position, 1.0, 1.0);
    gl_PointSize = pointsize;
}
