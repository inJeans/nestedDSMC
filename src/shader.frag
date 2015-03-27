#version 330 core

in vec4 pos;

out vec4 color;

uniform vec4 ourColor;

void main()
{
	color = 0.5 * ( pos + 1.0 );
}