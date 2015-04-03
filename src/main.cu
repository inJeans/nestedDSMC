//
//  main.c
//  nestedDSMC
//
//  Created by Christopher Watkins on 24/03/2015.
//
//

#include "main.cuh"

// Window dimensions
const GLuint WIDTH = 800, HEIGHT = 600;

float frameCount = 0;

// The MAIN function, from here we start the application and run the game loop
int main()
{
    std::cout << "Starting GLFW context, OpenGL 3.3" << std::endl;
    // Init GLFW
//    glfwInit();
    // Set all the required options for GLFW
//    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
//    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
//    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    
    // Create a GLFWwindow object that we can use for GLFW's functions
//    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL", NULL, NULL);
//    glfwMakeContextCurrent(window);
//    if (window == NULL)
//    {
//        std::cout << "Failed to create GLFW window" << std::endl;
//        glfwTerminate();
//        return -1;
//    }
    
    // Set the required callback functions
//    glfwSetKeyCallback(window, key_callback);
    
    // Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
//    glewExperimental = GL_TRUE;
    // Initialize GLEW to setup the OpenGL Function pointers
//    if (glewInit() != GLEW_OK)
//    {
//        std::cout << "Failed to initialize GLEW" << std::endl;
//        return -1;
//    }
    
//    int fbwidth, fbheight;
//    glfwGetFramebufferSize(window,
//                           &fbwidth,
//                           &fbheight) ;
    // Define the viewport dimensions
//    glViewport(0, 0, fbwidth, fbheight);
    
//	Shader ourShader("./src/shader.vert", "./src/shader.frag");
	
    // Set up vertex data (and buffer(s)) and attribute pointers
    GLfloat vertices[] = {
	   -0.5f, -0.5f, 0.0f,
        0.0f, -0.5f, 0.0f,
	    0.0f,  0.5f, 0.0f,
	    0.5f, -0.5f, 0.0f
    };
    GLuint indices[] = {  // Note that we start from 0!
        0, 1, 2,  // First Triangle
        1, 2, 3   // Second Triangle
    };
    GLuint VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    // Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s).
    glBindVertexArray(VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0); // Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound vertex buffer object so afterwards we can safely unbind
    
    glBindVertexArray(0); // Unbind VAO (it's always a good thing to unbind any buffer/array to prevent strange bugs), remember: do NOT unbind the EBO, keep it bound to this VAO
    
    struct cudaGraphicsResource *cudaVBOres;
    
    //Register VBO with CUDA
    cuGraphicsGLRegisterBuffer(cudaVBOres,
                               *VBO,
                               CU_GRAPHICS_REGISTER_FLAGS_NONE );
    
    // Uncommenting this call will result in wireframe polygons.
//    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
    // Game loop
//    while (!glfwWindowShouldClose(window))
//    {
        // Check if any events have been activiated (key pressed, mouse moved etc.) and call corresponding response functions
//        glfwPollEvents();
        
        // Render
        // Clear the colorbuffer
//        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
//        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
//        glClear(GL_COLOR_BUFFER_BIT);
        
//        ourShader.Use();
		
		// Update the uniform color
		GLfloat timeValue = glfwGetTime();
//		GLfloat greenValue = (sin(timeValue) / 2) + 0.5;
//		glUniform4f(glGetUniformLocation(ourShader.Program, "ourColor"), 0.0f, greenValue, 0.0f, 1.0f);
		
        //Map OpenGL buffer object for writing from CUDA
        float3 *d_pos;
        cudaGLMapBufferObject((void**)&d_pos, VBO);
		
        moveParticles<<< 1, 4 >>>( d_pos, timeValue, 4 );
        
        //Unmap buffer object
        cudaGLUnmapBufferObject(VBO);
        
//        glBindVertexArray(VAO);
//        glBindBuffer(GL_ARRAY_BUFFER, VBO);
//        glPointSize(10.0f);//set point size to 10 pixels
//        glDrawElements(GL_POINTS, 6, GL_UNSIGNED_INT, 0);
//        glDrawArrays(GL_POINTS, 0, 4);
//        glBindVertexArray(0);
//        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
//        computeFPS(window);
        
        // Swap the screen buffers
//        glfwSwapBuffers(window);
//    }
    
    /// Properly de-allocate all resources once they've outlived their purpose
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    // Terminate GLFW, clearing any resources allocated by GLFW.
//    glfwTerminate();
    return 0;
}

// Is called whenever a key is pressed/released via GLFW
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    std::cout << key << std::endl;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

void computeFPS( GLFWwindow* window )
{
    frameCount++;
    float avgFPS = frameCount / glfwGetTime();
    
    char fps[256];
    sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps", avgFPS);
    
    glfwSetWindowTitle( window, fps );
}