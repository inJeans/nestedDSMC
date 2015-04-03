#version 330 core

layout (location = 0) in vec3 position;

out vec4 pos;

void main()
{
	pos = vec4(position.x,
			   position.y,
			   position.z,
			   1.0f);
	
	gl_Position = pos;
}