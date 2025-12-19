#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

// --- ZMIENNE GLOBALNE ---
int width, height, channels;
int fbWidth, fbHeight;
unsigned char* imgData = nullptr;
float heightScale = 0.5f; // Skala wysokoœci

//zmienne do skali jasnoœci
float minVal = 255.0f;
float maxVal = 0.0f;

// Transformacje
float rotX = 45.0f;
float rotY = 0.0f;
float moveX = 0.0f; // Przesuniêcie w poziomie
float moveY = 0.0f; // Przesuniêcie w pionie
float zoom = 1.5f;

// Mysz
bool isRotating = false;
bool isPanning = false;
double lastMouseX, lastMouseY;




void key_callback( GLFWwindow* window, int key, int scancode, int action, int mods )
{
    if(key == GLFW_KEY_P && action == GLFW_PRESS)
    {
        cout << "Próba wykonania zrzutu ekranu" << endl;
        glfwGetFramebufferSize( window, &fbWidth, &fbHeight );
        vector<unsigned char> pixels( fbWidth * fbHeight * 3 );
        glReadPixels( 0, 0, fbWidth, fbHeight, GL_RGB, GL_UNSIGNED_BYTE, pixels.data() );
        stbi_flip_vertically_on_write( true );
        string filename = "screenshot_model.png";

        int result = stbi_write_png( filename.c_str(), fbWidth, fbHeight, 3, pixels.data(), fbWidth * 3 );

        if(result)
            cout << "Zapisano zrzut ekranu: " << filename << endl;
        else
            cout << "B³¹d podczas zapisu zrzutu ekranu" << endl;
    }
}

void scroll_callback( GLFWwindow* window, double xoffset, double yoffset )
{
    zoom += (float)yoffset * 0.1f;
    if(zoom < 0.1f) zoom = 0.1f;
}

void mouse_button_callback( GLFWwindow* window, int button, int action, int mods )
{
    if(action == GLFW_PRESS) {
        glfwGetCursorPos( window, &lastMouseX, &lastMouseY );
    }
    //LEWY
    if(button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if(action == GLFW_PRESS) isRotating = true;
        else if(action == GLFW_RELEASE) isRotating = false;
    }
    //PRAWY
    if(button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        if(action == GLFW_PRESS) isPanning = true;
        else if(action == GLFW_RELEASE) isPanning = false;
    }
}

void cursor_position_callback( GLFWwindow* window, double xpos, double ypos )
{
    double deltaX = xpos - lastMouseX;
    double deltaY = ypos - lastMouseY;
    //LEWY
    if(isRotating)
    {
        rotY += (float)deltaX * 0.5f;
        rotX += (float)deltaY * 0.5f;
    }
    //PRAWY
    if(isPanning)
    {
        float panSpeed = 1.0f / zoom;

        moveX += (float)deltaX * panSpeed;
        moveY -= (float)deltaY * panSpeed;
    }
    lastMouseX = xpos;
    lastMouseY = ypos;
}

// FUNKCJA KOLORUJ¥CA
float getHeightAndSetColor( int x, int z ) {
    if(x < 0 || x >= width || z < 0 || z >= height) return 0.0f;
    int index = (z * width + x) * channels;
    // wzór: (val - min) / (max - min) (NORMALIZACJA)
    float rawVal = (float)imgData[index];
    float range = maxVal - minVal;
    if(range <= 0.0f) range = 1.0f; //nie dzieliæ przez 0
    float normalized = (rawVal - minVal) / range;

    float y = (normalized * 255.0f) * heightScale;

    float peakThreshold = 0.87f; // od tego poziomu bia³y

    if(normalized >= peakThreshold) {
        glColor3f( 1.0f, 1.0f, 1.0f );
    }
    else {
        float greenGradientPos = normalized / peakThreshold;

        greenGradientPos = greenGradientPos * greenGradientPos;

        float darkR = 0.0f, darkG = 0.20f, darkB = 0.05f;

        float lightR = 0.3f, lightG = 0.85f, lightB = 0.15f;

        float finalR = darkR + greenGradientPos * (lightR - darkR);
        float finalG = darkG + greenGradientPos * (lightG - darkG);
        float finalB = darkB + greenGradientPos * (lightB - darkB);

        glColor3f( finalR, finalG, finalB );
    }

    return y;
}

void renderTerrain() {
    if(!imgData) return;
    glPushMatrix();
    glTranslatef( -width / 2.0f, 0.0f, -height / 2.0f );
    for(int z = 0; z < height - 1; z++) {
        glBegin( GL_TRIANGLE_STRIP );
        for(int x = 0; x < width; x++) {
            float y1 = getHeightAndSetColor( x, z );
            glVertex3f( (float)x, y1, (float)z );

            float y2 = getHeightAndSetColor( x, z + 1 );
            glVertex3f( (float)x, y2, (float)(z + 1) );
        }
        glEnd();
    }
    glPopMatrix();
}

void repairEdges() {
    int margin = 5;
    if(!imgData || width < 2 * margin || height < 2 * margin) return;

    cout << "naprawianie krawedzi (margines: " << margin << " px)" << endl;

    for(int x = 0; x < width; x++) {
        int idxTopSafe = (margin * width + x) * channels;
        int idxBotSafe = ((height - 1 - margin) * width + x) * channels;

        unsigned char topVal = imgData[idxTopSafe];
        unsigned char botVal = imgData[idxBotSafe];
        for(int k = 0; k < margin; k++) {
            int idxTop = (k * width + x) * channels;
            imgData[idxTop] = topVal;
            int idxBot = ((height - 1 - k) * width + x) * channels;
            imgData[idxBot] = botVal;
        }
    }
    for(int z = 0; z < height; z++) {
        int idxLeftSafe = (z * width + margin) * channels;
        int idxRightSafe = (z * width + (width - 1 - margin)) * channels;

        unsigned char leftVal = imgData[idxLeftSafe];
        unsigned char rightVal = imgData[idxRightSafe];

        for(int k = 0; k < margin; k++) {
            int idxLeft = (z * width + k) * channels;
            imgData[idxLeft] = leftVal;
            int idxRight = (z * width + (width - 1 - k)) * channels;
            imgData[idxRight] = rightVal;
        }
    }
}

int main( void )
{
    GLFWwindow* window;

    if(!glfwInit()) return -1;

    int windowWidth = 1280;
    int windowHeight = 720;
    window = glfwCreateWindow( 1280, 720, "Model NMT - KOLOROWY", NULL, NULL );
    if(!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent( window );

    glfwSetScrollCallback( window, scroll_callback );
    glfwSetMouseButtonCallback( window, mouse_button_callback );
    glfwSetCursorPosCallback( window, cursor_position_callback );
    glfwSetKeyCallback( window, key_callback );

    imgData = stbi_load( "terrain.png", &width, &height, &channels, 0 );


    if(imgData) {
        cout << "Wczytano model: " << width << "x" << height << endl;
        repairEdges();
        float localMin = 255.0f;
        float localMax = 0.0f;

        for(int i = 0; i < width * height; i++) {
            unsigned char val = imgData[i * channels];
            if((float)val > localMax) localMax = (float)val;
            if((float)val < localMin) localMin = (float)val;
        }
        minVal = localMin;
        maxVal = localMax;
    }
    else {
        cout << "B³¹d - Brak pliku terrain.png!" << endl;
    }

    glEnable( GL_DEPTH_TEST );

    while(!glfwWindowShouldClose( window ))
    {
        // t³o
        glClearColor( 0.53f, 0.81f, 0.92f, 1.0f );
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        // kamera
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();

        glfwGetWindowSize( window, &windowWidth, &windowHeight );
        float aspect = 1280.0f / 720.0f;
        float viewSize = 500.0f / zoom;

        glOrtho( -viewSize * aspect, viewSize * aspect, -viewSize, viewSize, -1000, 1000 );

        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();

        // transformacje - rotacja i przesuniêcie
        glTranslatef( moveX, moveY, 0.0f );
        glRotatef( rotX, 1.0f, 0.0f, 0.0f );
        glRotatef( rotY, 0.0f, 1.0f, 0.0f );

        renderTerrain();

        glfwSwapBuffers( window );
        glfwPollEvents();
    }

    if(imgData) stbi_image_free( imgData );
    glfwTerminate();
    return 0;
}