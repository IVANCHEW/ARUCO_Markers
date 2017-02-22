#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <thread>
#include <math.h>
#include <cmath>
#include "ConfigFile.cpp"
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

class DataManagement
{
	private:
	
		cv::Mat tvec;
		cv::Mat rvec;
		cv::Mat frame;
		bool transformation_ready = false;
		bool image_ready = false;
		
	public:
	
		void loadTransform(cv::Mat t, cv::Mat r);
		void loadFrame(cv::Mat f);
		
		void getTransform(cv::Mat& t, cv::Mat& r);
		bool getFrame(cv::Mat& f);
		
		bool statusTransform();
};

void DataManagement::loadTransform(cv::Mat t, cv::Mat r){
	tvec = t;
	rvec = r;
	transformation_ready = true;
}

void DataManagement::loadFrame(cv::Mat f){
		frame = f.clone();
		image_ready = true;
}

void DataManagement::getTransform(cv::Mat& t, cv::Mat& r){
	t = tvec;
	r = rvec;
}

bool DataManagement::getFrame(cv::Mat& f){
	if (image_ready){
		f = frame.clone();
		image_ready = false;
		return true;
	}
	else
		return false;
}

bool DataManagement::statusTransform(){
	return transformation_ready;
}

// For Data Management 
DataManagement dm;

// For Debugging
bool debug_rendering_ = false;
//~ bool debug_rendering_ = true;

//~ bool debug_calibration_ = false;
bool debug_calibration_ = true;

// For Texturing
GLuint textures[2];
cv::Mat mat_vertex;
cv::Mat obj_tex_mat;

// For Camera Parameters
cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
cv::Mat camera_matrix;
cv::Mat dist_coeffs;
std::string calibration_file = "calibration3";
std::string calibration_extension = ".jpg";
float calibration_square_size = 0.022;

// For ARUCO
int target_id = 1;
float aruco_size = 0.08/2;

// For Camera Frame Rendering
int window_height = 480;
int window_width = 640;
float clipping_length;
float focal_length;
float frame_height;
float frame_width;
float camera_distance = 10;
int framecount = 0;
bool frame_ready = false;

float x_correction = 0.0;
float y_correction = 0.0;
float z_correction = 0.0;
float f_correction = 0.0;
float correct_increment = 0.001;

// For Virtual Object Rendering
bool extract_normals = true;
//~ bool extract_normals = false;

//~ int vertex_type = 3;
int vertex_type = 4;

bool extract_colors = false;

//~ ConfigFile cf("model/model2.txt");
//~ ConfigFile cf("model/simple_cube.txt");
//~ ConfigFile cf("model/short_ws.txt");
ConfigFile cf("model/heli_full.txt");
std::vector<std::vector<float> > vertex;
std::vector<std::vector<float> > normals;
std::vector<std::vector<float> > vertex_color;
std::vector<std::vector<int> > vertex_index;

ConfigFile cf_texture("model/heli_full_texture.txt");
std::vector<std::vector<float> > tex_vertex;
std::vector<std::vector<float> > tex_normals;
std::vector<std::vector<int> > tex_vertex_index;

void rotateImage(cv::Mat& image, double angle){
	std::cout << "Original Image size: " << image.size() << std::endl;
	cv::Mat R_Matrix;
	cv::Point2f center;
	center.y = image.rows/2;
	center.x = image.cols/2;
	R_Matrix = cv::getRotationMatrix2D(center, angle, 1.0);
	cv::warpAffine(image, image, R_Matrix, image.size());
	std::cout << "Transformed Image size: " << image.size() << std::endl;
}

static void calcBoardCornerPositions(cv::Size boardSize, float squareSize, std::vector<cv::Point3f>& corners, std::string patternType){
    corners.clear();

    if (patternType == "CHESSBOARD" || patternType == "CIRCLES_GRID"){
			for( int i = 0; i < boardSize.height; ++i )
					for( int j = 0; j < boardSize.width; ++j )
							corners.push_back(cv::Point3f(float( j*squareSize ), float( i*squareSize ), 0));
		}

    else if (patternType == "ASYMMETRIC_CIRCLES_GRID"){
			for( int i = 0; i < boardSize.height; i++ )
					for( int j = 0; j < boardSize.width; j++ )
							corners.push_back(cv::Point3f(float((2*j + i % 2)*squareSize), float(i*squareSize), 0));
		}

}

//Future work: Add refinement to initial corner detection and post-calibration rms error
void calibrateCamera(cv::Mat& camera_matrix, cv::Mat& dist_coeffs, bool show_board_image){
	
	// Functional Variables
	std::vector<std::vector<cv::Point2f> > image_points;
	std::vector<std::vector<cv::Point3f> > object_points(1);
	cv::Size board_size;
	cv::Size calibration_image_size;
	std::vector<cv::Mat> rvecs;
	std::vector<cv::Mat> tvecs;
	cv::Mat first_board_image;
	first_board_image = cv::imread(calibration_file + "/0" + calibration_extension, CV_LOAD_IMAGE_COLOR);   
	calibration_image_size.width = first_board_image.cols;
	calibration_image_size.height = first_board_image.rows;
	if (debug_calibration_)
		std::cout << "Calibration image size: " << first_board_image.cols << " " << first_board_image.rows << std::endl;
	
	// Contextual Parameters
	board_size.width = 9;
	board_size.height = 6;	
	std::string calibration_pattern = "CHESSBOARD"; // Available Pattern Types: "CHESSBOARD", "CIRCLES_GRID", "ASYMMETRIC_CIRCLES_GRID"
	
	// NOTE: Maximum of 10 calibration images
	for(int i = 0 ; i < 10 ; i++){
		
		// STEP 1: Load Calibration Image
		std::stringstream convert;
		convert << i;
		std::string s;
		s = convert.str();
		cv::Mat board_image;
		board_image = cv::imread(calibration_file + "/" + s + "" + calibration_extension, CV_LOAD_IMAGE_COLOR);   
		
		if (! board_image.data){
			std::cout <<  "Total Number of calibration images: " << i << std::endl ;
			break;
		}
		
		// STEP 2: Search for calibration image keypoints
		std::vector<cv::Point2f> point_buf;
		bool found;
		cv::Mat gray;
		cv::cvtColor(board_image, gray, cv::COLOR_BGR2GRAY);
		found = cv::findChessboardCorners(gray, board_size, point_buf);
		if (found){			
			if (show_board_image){
				cv::drawChessboardCorners(board_image, board_size, cv::Mat(point_buf), found );
				cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
				cv::imshow( "Display window", board_image );
				cv::waitKey(0);
			}
			image_points.push_back(point_buf);
		}
		else{
			std::cout << "Could not detect key points for frame: " << i << std::endl;
		}
	}
	
	std::cout << "Number of patterns found: " << image_points.size() << std::endl;
	
	// STEP 3: Perform calibration if at least 5 good images are available
	if (image_points.size() >= 5){
		calcBoardCornerPositions(board_size, calibration_square_size, object_points[0], calibration_pattern);
		object_points.resize(image_points.size(),object_points[0]);
		double rms = calibrateCamera(object_points, image_points, calibration_image_size, camera_matrix, dist_coeffs, rvecs, tvecs);
		std::cout << "Reprojection-Error from calibration: " << rms << std::endl;
	}
	
}

bool arucoPoseEstimation(cv::Mat& input_image, int id, cv::Mat& tvec, cv::Mat& rvec, cv::Mat& mtx, cv::Mat& dist, bool draw_axis){
	// Contextual Parameters
	//~ std::cout << "Pose estimation called" << std::endl;
	float aruco_square_size = aruco_size*2;
	bool marker_found = false;
	std::vector< int > marker_ids;
	std::vector< std::vector<cv::Point2f> > marker_corners, rejected_candidates;
	cv::Mat gray;
	
	cv::cvtColor(input_image, gray, cv::COLOR_BGR2GRAY);
	cv::aruco::detectMarkers(gray, dictionary, marker_corners, marker_ids);	
	
	if (marker_ids.size() > 0){
		for (int i = 0 ; i < marker_ids.size() ; i++){
			//~ std::cout << "Marker IDs found: " << marker_ids[i] << std::endl;
			if (marker_ids[i] == id){
				std::vector< std::vector<cv::Point2f> > single_corner(1);
				single_corner[0] = marker_corners[i];
				cv::aruco::estimatePoseSingleMarkers(single_corner, aruco_square_size, mtx, dist, rvec, tvec);
				if (draw_axis){
					cv::aruco::drawDetectedMarkers(input_image, marker_corners, marker_ids);
					cv::aruco::drawAxis(input_image, mtx, dist, rvec, tvec, aruco_square_size/2);
				}
				//~ std::cout << "Marker found : aruco pose estimation" << std::endl;
				marker_found = true;
			}
		}
	}
	else{
		//~ std::cout << "No markers detected" << std::endl;
	}
	
	return marker_found;
}

void changeSize(int w, int h) {

	std::cout <<std::endl<<std::endl << "Change Size Callback" << std::endl;
	std::cout << "Size detected: " << w << " x " << h << std::endl;
	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if (h == 0)
		h = 1;

	float ratio =  w * 1.0 / h;
	
	// Use the Projection Matrix
	glMatrixMode(GL_PROJECTION);

	// Reset Matrix
	glLoadIdentity();

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

	// Set the correct perspective.
	// Original FOV is 45 deg.
	std::cout << "Aspect ratio: " << ratio << std::endl;
	gluPerspective(45.0f, ratio, 0.1f, 100.0f);

	// Get Back to the Modelview
	glMatrixMode(GL_MODELVIEW);
	
	//std::cout << "Size checked" << std::endl;
}

void compute_camera_frame_size(){
	float width = mat_vertex.cols;
	float height = mat_vertex.rows;
	float pixel_ratio = camera_distance/focal_length;
	frame_height = height * pixel_ratio / 2;
	frame_width = width * pixel_ratio / 2;
}

void processNormalKeys(unsigned char key, int x, int y) {
	//~ std::cout << "key press: " << key << std::endl;
	if (key == 'q'){
		x_correction = x_correction + correct_increment;
		std::cout << "X_correct: " << x_correction << std::endl;
	}
	if (key == 'a'){
		x_correction = x_correction - correct_increment;
		std::cout << "X_correct: " << x_correction << std::endl;
	}
	if (key == 'w'){
		y_correction = y_correction + correct_increment;
		std::cout << "Y_correct: " << y_correction << std::endl;
	}
	if (key == 's'){
		y_correction = y_correction - correct_increment;
		std::cout << "Y_correct: " << y_correction << std::endl;
	}
	if (key == 'e'){
		z_correction = z_correction + correct_increment;
		std::cout << "Z_correct: " << z_correction << std::endl;
	}
	if (key == 'd'){
		z_correction = z_correction - correct_increment;
		std::cout << "Z_correct: " << z_correction << std::endl;
	}
	if (key == 'r'){
		focal_length++;
		std::cout << "focal_length: " << focal_length << std::endl;
		compute_camera_frame_size();
	}
	if (key == 'f'){
		focal_length--;
		std::cout << "focal_length: " << focal_length << std::endl;
		compute_camera_frame_size();
	}
}

void init_texture(void){
	
	std::cout << std::endl << "Initiating Vertex" << std::endl;
	
	mat_vertex = cv::imread("test/0.png");  
	
	float width = mat_vertex.cols;
	float height = mat_vertex.rows;
	float aspect_ratio = width/height;
	std::cout << aspect_ratio << std::endl;
	clipping_length = (1.0 / 2.0) * ( 1.0 - 1.0/ aspect_ratio);
	
	float pixel_ratio = camera_distance/focal_length;
	frame_height = height * pixel_ratio / 2;
	frame_width = width * pixel_ratio / 2;

	std::cout << "Pixelratio: " << pixel_ratio <<  std::endl;
	std::cout << "Frame: " << frame_height << " " << frame_width << std::endl;
	
	cv::flip(mat_vertex, mat_vertex, 0);
	glGenTextures(2, textures);
	
	glBindTexture(GL_TEXTURE_2D, textures[0]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mat_vertex.cols, mat_vertex.rows,  0, GL_BGR, GL_UNSIGNED_BYTE, mat_vertex.ptr());
	
	// For object texture
	obj_tex_mat = cv::imread("texture/AFTC.png", -1);
	//~ rotateImage(obj_tex_mat, 90);
	glBindTexture(GL_TEXTURE_2D, textures[1]);	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, obj_tex_mat.cols, obj_tex_mat.rows,  0, GL_BGRA, GL_UNSIGNED_BYTE, obj_tex_mat.ptr());
	
}

void update_texture(void){
	frame_ready = dm.getFrame(mat_vertex);
	if (frame_ready){
		glBindTexture(GL_TEXTURE_2D, textures[0]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mat_vertex.cols, mat_vertex.rows,  0, GL_BGR, GL_UNSIGNED_BYTE, mat_vertex.ptr());
	}
}

void init_lighting(void){
	GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat mat_shininess[] = { 50.0 };
	GLfloat light_position[] = { 0.0, 1.0, 0.0, 0.0 };
	glClearColor (0.0, 0.0, 0.0, 0.0);
	glShadeModel (GL_SMOOTH);

	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
}

void renderScene(void) {
	
	// Clear Color and Depth Buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Reset transformations
	glLoadIdentity();
	
	// Set the camera
	gluLookAt(	0.0f, 0.0f, 0.0f,
				0.0f, 0.0f,  1.0f,
				0.0f, 1.0f,  0.0f);
					
	// For Camera Background
	update_texture();
	
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, textures[0]);
	//~ glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glBegin(GL_QUADS);

	 glTexCoord2f(0.0, 1.0);
	 glVertex3f(-frame_width, -frame_height, camera_distance);
	 glTexCoord2f(0.0, 0.0);
	 glVertex3f(-frame_width, frame_height, camera_distance);
	 glTexCoord2f(1.0, 0.0);
	 glVertex3f(frame_width, frame_height, camera_distance);
	 glTexCoord2f(1.0, 1.0);
	 glVertex3f(frame_width, -frame_height, camera_distance);

	glEnd();
	
	glBindTexture(GL_TEXTURE_2D, 0); 
	glDisable(GL_TEXTURE_2D);
	
	// For Virtual Object Transformation
	if (dm.statusTransform()){		
		cv::Mat tvec;
		cv::Mat rvec;
		dm.getTransform(tvec, rvec);
		//~ std::cout << tvec << std::endl;
		float a = rvec.at<double>(0);
		float b = rvec.at<double>(1);
		float c = rvec.at<double>(2);
		float theta = sqrt(pow(a,2) + pow(b,2) + pow(c,2));
		glTranslatef(tvec.at<double>(0) + x_correction, -tvec.at<double>(1) - y_correction, tvec.at<double>(2) + z_correction);	
		glRotatef(theta*180/M_PI, -a/theta, b/theta, -c/theta);

	}
	
	//~ // For Virtual Object Lighting
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	
	//~ // For Virtual Object Rendering
	
	//~ // For 4-sided cube
	//~ glBegin(GL_QUADS); // of the color cube
		//~ glColor3f(0.0f, 1.0f, 0.0f); // green
		//~ glVertex3f(aruco_size, -aruco_size, 0);
		//~ glVertex3f(-aruco_size, -aruco_size, 0);
	  //~ glVertex3f(-aruco_size, aruco_size, 0);
	  //~ glVertex3f(aruco_size, aruco_size, 0);
	  
	  //~ glColor3f(1.0f, 0.0f, 0.0f); // red
		//~ glVertex3f(aruco_size, aruco_size, 0);
		//~ glVertex3f(aruco_size, -aruco_size, 0);
	  //~ glVertex3f(aruco_size, -aruco_size, aruco_size*2);
	  //~ glVertex3f(aruco_size, aruco_size, aruco_size*2);
	  
	  //~ glColor3f(0.0f, 0.0f, 1.0f); // blue
		//~ glVertex3f(aruco_size, aruco_size, aruco_size*2);
		//~ glVertex3f(-aruco_size, aruco_size, aruco_size*2);
	  //~ glVertex3f(-aruco_size, -aruco_size, aruco_size*2);
	  //~ glVertex3f(aruco_size, -aruco_size, aruco_size*2);
	  
	  //~ glColor3f(1.0f, 0.5f, 0.5f); // orange
		//~ glVertex3f(-aruco_size, -aruco_size, aruco_size*2);
		//~ glVertex3f(-aruco_size, aruco_size, aruco_size*2);
	  //~ glVertex3f(-aruco_size, aruco_size, 0);
	  //~ glVertex3f(-aruco_size, -aruco_size, 0);
	//~ glEnd();

	// For model rendered from .txt file
	int scount = 0;
	if (vertex_type==3){
		glBegin(GL_TRIANGLES);
	}else if (vertex_type==4){
		glBegin(GL_QUADS);
	}
		for (int i = 0 ; i < vertex_index.size() ; i++){
			for (int j = 0 ; j < vertex_type ; j++){
				if (extract_normals)
					glNormal3f(normals[vertex_index[i][j]][0],normals[vertex_index[i][j]][1],normals[vertex_index[i][j]][2]);
					
				glVertex3f(vertex[vertex_index[i][j]][0],vertex[vertex_index[i][j]][1], vertex[vertex_index[i][j]][2]);
			}
		}
	glEnd();
	
	// For disabling lighting
	glDisable(GL_COLOR_MATERIAL);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	
	// For logo texture
	glEnable(GL_TEXTURE_2D);
	
	//~ glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	//~ glTexEnvf(GL_TEXTURE_ENV, GL_SRC1_ALPHA, GL_DECAL);
	glBindTexture(GL_TEXTURE_2D, textures[1]);	
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	
	glBegin(GL_QUADS);

	 glTexCoord2f(0.0, 0.0);
	 glNormal3f(tex_normals[tex_vertex_index[0][0]][0],tex_normals[tex_vertex_index[0][0]][1],tex_normals[tex_vertex_index[0][0]][2]);
	 glVertex3f(tex_vertex[tex_vertex_index[0][0]][0],tex_vertex[tex_vertex_index[0][0]][1], tex_vertex[tex_vertex_index[0][0]][2]);
	 glTexCoord2f(1.0, 0.0);
	 glNormal3f(tex_normals[tex_vertex_index[0][1]][0],tex_normals[tex_vertex_index[0][1]][1],tex_normals[tex_vertex_index[0][1]][2]);
	 glVertex3f(tex_vertex[tex_vertex_index[0][1]][0],tex_vertex[tex_vertex_index[0][1]][1], tex_vertex[tex_vertex_index[0][1]][2]);
	 glTexCoord2f(1.0, 1.0);
	 glNormal3f(tex_normals[tex_vertex_index[0][2]][0],tex_normals[tex_vertex_index[0][2]][1],tex_normals[tex_vertex_index[0][2]][2]);
	 glVertex3f(tex_vertex[tex_vertex_index[0][2]][0],tex_vertex[tex_vertex_index[0][2]][1], tex_vertex[tex_vertex_index[0][2]][2]);
	 glTexCoord2f(0.0, 1.0);
	 glNormal3f(tex_normals[tex_vertex_index[0][3]][0],tex_normals[tex_vertex_index[0][3]][1],tex_normals[tex_vertex_index[0][3]][2]);
	 glVertex3f(tex_vertex[tex_vertex_index[0][3]][0],tex_vertex[tex_vertex_index[0][3]][1], tex_vertex[tex_vertex_index[0][3]][2]);

	glEnd();
	
	glBindTexture(GL_TEXTURE_2D, 0); 
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_BLEND);
	
	glutSwapBuffers();
	
}

// Check Mat type (from stackOverflow)
std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

// THREAD FUNCTIONS
void *start_gl_main(void *threadid){
	long tid;
  tid = (long)threadid;
  std::cout << "Start gl main thread id : " << tid << std::endl;
	glutMainLoop();
	pthread_exit(NULL);
}

void *start_cv_main(void *threadid){
	cv::VideoCapture inputVideo;
	inputVideo.open(0);
	bool marker_found;
	while (inputVideo.grab()) {
		cv::Mat image;
		cv::Mat unDistort;
		cv::Mat rvec;
		cv::Mat tvec;
		inputVideo.retrieve(image);
		
		marker_found = false;
		marker_found = arucoPoseEstimation(image, target_id, tvec, rvec, camera_matrix, dist_coeffs, true);
		cv::undistort(image, unDistort, camera_matrix, dist_coeffs);
		//~ std::cout << "Camera image size: " << image.size() << std::endl;
		dm.loadFrame(unDistort);
		//~ dm.loadFrame(image);
		if (marker_found==true){
			//~ std::cout << "Marker found" << std::endl;
			//~ std::cout << tvec << std::endl;
			//~ std::cout << tvec.at<double>(0) << std::endl;
			dm.loadTransform(tvec, rvec);			
		}
		//~ cv::imshow("out", image);
		//~ char key = (char) cv::waitKey(1);
	  //~ if (key == 27){
			//~ break;
			//~ inputVideo.release();
			//~ pthread_exit(NULL);
		//~ }
	}
}

int main( int argc, char** argv )
{
	// CALIBRATE CAMERA
	camera_matrix = cv::Mat::eye(3, 3, CV_64F);
	dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
	calibrateCamera(camera_matrix, dist_coeffs, false);
	focal_length = camera_matrix.at<double>(0,0);
	//~ focal_length = 800;
	
	// LOAD GL MODEL
	if (extract_normals){
		cf.toggleExtractNormals(true);		
		cf.extractKeys();
		cf.getNormals(normals);
		cf.getVertex(vertex);
		cf.getIndex(vertex_index);
	}
	else{
		cf.extractKeys();
		cf.getVertex(vertex);
		cf.getIndex(vertex_index);
	}
	
	// LOAD GL TEXTURE MODEL
	if (extract_normals){
		cf_texture.toggleExtractNormals(true);
		cf_texture.extractKeys();
		cf_texture.getNormals(tex_normals);
		cf_texture.getVertex(tex_vertex);
		cf_texture.getIndex(tex_vertex_index);
	}
	
	std::cout << std::endl << "Number of vertex: " << vertex.size() << std::endl;
	std::cout << "Number of faces: " << vertex_index.size() << std::endl;
	
	// DEBUGGING
	if (debug_calibration_){
		std::cout << std::endl << "Calibration Matrix: " << std::endl << std::setprecision(5);
		for (int i=0 ; i<3 ; i++){
			std::cout << "[ " ;
			for (int j=0 ; j<3 ; j++)
				std::cout << camera_matrix.at<double>(i,j) << " ";
			std::cout << "]" << std::endl;
			}
		std::cout << std::endl << "Focal Length: " << focal_length << std::endl;
	}
	
	if (debug_rendering_){
		std::cout << "Size of vertex: " << vertex.size() << std::endl;
		for (int i = 0 ; i < vertex.size() ; i++)
			std::cout << vertex[i][0] << " , " << vertex[i][1] << " , " << vertex[i][2] << std::endl;
		std::cout << "Normals: " << std::endl;
		if (extract_normals){
			for (int i = 0 ; i < vertex.size() ; i++)
				std::cout << normals[i][0] << " , " << normals[i][1] << " , " << normals[i][2] << std::endl;
		}
		else{
			std::cout << "No normal extraction" << std::endl;
		}
	}

	//~ ConfigFile cf2("model/color.txt");
	//~ cf2.getVertex(vertex_color);

	// GLUT INITIALISATION
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(window_width,window_height);
	glutCreateWindow("Lighthouse3D- GLUT Tutorial");
	init_texture();
	init_lighting();
	
	//Load Model
	std::cout << "Try loading model" << std::endl;

	// GLUT CALLBACK REGISTRATION
	glutDisplayFunc(renderScene);
	glutReshapeFunc(changeSize);
	glutIdleFunc(renderScene);
	glutKeyboardFunc(processNormalKeys);

	// THREAD INITIALISATION	
	pthread_t thread[2];
	int threadError;
	int i=0;

	// START GL THREAD
	threadError = pthread_create(&thread[i], NULL, start_gl_main, (void *)i);

	if (threadError){
		std::cout << "Error:unable to create thread," << threadError << std::endl;
		exit(-1);
	}

	// START CV THREAD
	i++;
	threadError = pthread_create(&thread[i], NULL, start_cv_main, (void *)i);

	if (threadError){
		std::cout << "Error:unable to create thread," << threadError << std::endl;
		exit(-1);
	}
	
	pthread_exit(NULL);
	
	return 0;
}

