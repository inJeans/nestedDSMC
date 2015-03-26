//
//  shader.hpp
//  nestedDSMC
//
//  Created by Christopher Watkins on 26/03/2015.
//
//

#ifndef __nestedDSMC__shader__
#define __nestedDSMC__shader__

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
using namespace std;

#include <GL/glew.h> // Include glew to get all the required OpenGL headers

class Shader
{
public:
	// The program ID
	GLuint Program;
	// Constructor reads and builds the shader
	Shader(const GLchar* vertexSourcePath, const GLchar* fragmentSourcePath);
	// Use the program
	void Use();
};

#endif /* defined(__nestedDSMC__shader__) */
