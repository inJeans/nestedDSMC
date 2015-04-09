//
//  camera.hpp
//  nestedDSMC
//
//  Created by Christopher Watkins on 26/03/2015.
//
//

#ifndef __nestedDSMC__camera__
#define __nestedDSMC__camera__

// Std. Includes
#include <vector>

// GL Includes
#include <GL/glew.h> // Include glew to get all the required OpenGL headers
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Defines several possible options for camera movement.
// Used as abstraction to stay away from window-system specific input methods
enum Camera_Movement {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT
};

// Default camera values
const GLfloat THETA      =  90.0f;
const GLfloat PHI        =  0.0f;
const GLfloat RADIUS     =  5.0f;
const GLfloat SPEED      =  10.0f;
const GLfloat SENSITIVTY =  0.25f;
const GLfloat ZOOM       =  45.0f;

const glm::vec3 POS    = glm::vec3(0.0f, 0.0f, 0.0f);
const glm::vec3 UP     = glm::vec3(0.0f, 1.0f, 0.0f);
const glm::vec3 FRONT  = glm::vec3(0.0f, 0.0f, -1.0f);
const glm::vec3 CENTER = glm::vec3(0.0f, 0.0f, 0.0f);


// An abstract camera class that processes input and calculates
// the corresponding Eular Angles, Vectors and Matrices for use in OpenGL
class Camera
{
public:
	// Camera Attributes
	glm::vec3 Position;
	glm::vec3 Front;
	glm::vec3 Up;
	glm::vec3 Right;
	glm::vec3 WorldUp;
	glm::vec3 Center;
	// Eular Angles
	GLfloat Radius;
	GLfloat Theta;
	GLfloat Phi;
	// Camera options
	GLfloat MovementSpeed;
	GLfloat MouseSensitivity;
	GLfloat Zoom;
	
	// Constructor with vectors
	Camera(glm::vec3 position = POS,
		   glm::vec3 up = UP,
		   glm::vec3 center = CENTER,
		   GLfloat radius = RADIUS,
		   GLfloat theta = THETA,
		   GLfloat phi = PHI );
	
	// Constructor with scalar values
	Camera(GLfloat posX = POS.x,
		   GLfloat posY = POS.y,
		   GLfloat posZ = POS.z,
		   GLfloat upX = UP.x,
		   GLfloat upY = UP.x,
		   GLfloat upZ = UP.z,
		   GLfloat centerX = CENTER.x,
		   GLfloat centerY = CENTER.y,
		   GLfloat centerZ = CENTER.z,
		   GLfloat radius = RADIUS,
		   GLfloat theta = THETA,
		   GLfloat phi = PHI);
	
	// Returns the view matrix calculated using Eular Angles and the LookAt Matrix
	glm::mat4 GetViewMatrix();
	
	// Processes input received from any keyboard-like input system.
	// Accepts input parameter in the form of camera defined ENUM
	// (to abstract it from windowing systems)
	void ProcessKeyboard(Camera_Movement direction,
						 GLfloat deltaTime);
	
//	// Processes input received from a mouse input system.
//	// Expects the offset value in both the x and y direction.
//	void ProcessMouseMovement(GLfloat xoffset,
//							  GLfloat yoffset,
//							  GLboolean constrainPitch = true);
	
	// Processes input received from a mouse scroll-wheel event.
	// Only requires input on the vertical wheel-axis
	void ProcessMouseScroll(GLfloat yoffset);
	
private:
	// Calculates the front vector from the Camera's (updated) Eular Angles
	void updateCameraVectors();
};

#endif /* defined(__nestedDSMC__camera__) */