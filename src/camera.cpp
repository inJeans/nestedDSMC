//
//  camera.cpp
//  nestedDSMC
//
//  Created by Christopher Watkins on 08/04/2015.
//
//

#include "camera.hpp"
#include <iostream>

Camera::Camera(glm::vec3 position,
			   glm::vec3 up,
			   glm::vec3 center,
			   GLfloat radius,
			   GLfloat theta,
			   GLfloat phi)
			  :	Front(FRONT),
				MovementSpeed(SPEED),
				MouseSensitivity(SENSITIVTY),
				Zoom(ZOOM)
{
	this->Position = position;
	this->WorldUp  = up;
	this->Center   = center;
	this->Radius   = radius;
	this->Theta    = theta;
	this->Phi      = phi;
	this->updateCameraVectors();
}

// Constructor with scalar values
Camera::Camera(GLfloat posX,
			   GLfloat posY,
			   GLfloat posZ,
			   GLfloat upX,
			   GLfloat upY,
			   GLfloat upZ,
			   GLfloat centerX,
			   GLfloat centerY,
			   GLfloat centerZ,
			   GLfloat radius,
			   GLfloat theta,
			   GLfloat phi)
			  :	Front(FRONT),
				MovementSpeed(SPEED),
				MouseSensitivity(SENSITIVTY),
				Zoom(ZOOM)
{
	this->Position = glm::vec3(posX, posY, posZ);
	this->WorldUp  = glm::vec3(upX, upY, upZ);
	this->Center   = glm::vec3(centerX, centerY, centerZ);
	this->Radius   = radius;
	this->Theta    = theta;
	this->Phi      = phi;
	this->updateCameraVectors();
}

// Returns the view matrix calculated using Eular Angles and the LookAt Matrix
glm::mat4 Camera::GetViewMatrix()
{
	return glm::lookAt(this->Position,
					   this->Center,
					   this->Up);
}

// Processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
void Camera::ProcessKeyboard(Camera_Movement direction,
							 GLfloat deltaTime)
{
	GLfloat velocity = this->MovementSpeed * deltaTime;

	if (direction == FORWARD)
	{
		this->Theta -= velocity;
	}
	else if (direction == BACKWARD)
	{
		this->Theta += velocity;
	}
	else if (direction == LEFT)
	{
		this->Phi -= velocity;
	}
	else if (direction == RIGHT)
	{
		this->Phi += velocity;
	}
	
	// Update Front, Right and Up Vectors using the updated Eular angles
	this->updateCameraVectors();
	
	return;
}

//// Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
//void Camera::ProcessMouseMovement(GLfloat xoffset,
//								  GLfloat yoffset,
//								  GLboolean constrainPitch)
//{
//	xoffset *= this->MouseSensitivity;
//	yoffset *= this->MouseSensitivity;
//	
//	this->Phi   += xoffset;
//	this->Theta += yoffset;
//	
//	// Update Front, Right and Up Vectors using the updated Eular angles
//	this->updateCameraVectors();
//	
//	return;
//}

// Processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
void Camera::ProcessMouseScroll(GLfloat yoffset)
{
	this->Radius -= yoffset;
	this->updateCameraVectors();
	
	return;
}

// Calculates the front vector from the Camera's (updated) Eular Angles
void Camera::updateCameraVectors()
{
	// Calculate the new Front vector
	glm::vec3 front;
	front.x = sin(glm::radians(this->Phi)) * sin(glm::radians(this->Theta));
	front.y = cos(glm::radians(this->Theta));
	front.z = cos(glm::radians(this->Phi)) * sin(glm::radians(this->Theta));
	this->Front = glm::normalize(front);
	
	this->Position = this->Radius * this->Front;
	
	// Also re-calculate the Right and Up vector
	this->Right = glm::normalize(glm::cross(this->Front,
											this->WorldUp));
	this->Up    = glm::normalize(glm::cross(this->Right,
											this->Front));
	
	std::cout << "pos = { " << this->Position.x << ", " << this->Position.y << ", " << this->Position.z << " }" << std::endl;
	
	return;
}