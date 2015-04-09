#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec4 colour;

uniform float Tx;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec4 pos;
out vec4 col;

void main()
{
	pos = vec4(position.x/Tx,
			   position.y/Tx,
			   position.z/Tx,
			   1.0f);
	
	gl_Position = projection * view * model * pos;
	
	col = colour;
}