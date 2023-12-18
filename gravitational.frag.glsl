#version 330 core

in vec3 p;
in vec3 b;

void main() {
	float lineWidth = 0.01;
	float f_closest_edge = min(b.x, min(b.y, b.z)); // see to which edge this pixel is the closest
	float f_width = fwidth(f_closest_edge); // calculate derivative (divide lineWidth by this to have the line width constant in screen-space)
	float f_alpha = smoothstep(lineWidth, lineWidth + f_width, f_closest_edge); // calculate alpha
	gl_FragColor = vec4(vec3((1.0 - f_alpha) * 0.2), 1.0);
	//gl_FragColor = vec4(p * ((1.0 - f_alpha) * 0.2) + 0.1, 1.0);
}
