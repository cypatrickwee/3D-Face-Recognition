#ifndef PCASVM_H
#define PCASVM_H

//vtk libraries
#define vtkRenderingCore_AUTOINIT 4(vtkInteractionStyle,vtkRenderingFreeType,vtkRenderingFreeTypeOpenGL,vtkRenderingOpenGL)
#define vtkRenderingVolume_AUTOINIT 1(vtkRenderingVolumeOpenGL)
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>

//standard libraries
#include <ctime>
#include <iostream>
#include <list>
#include <vector>
#include <memory>
#include <functional>
#include <utility>
#include <fstream>
#include <sstream>
#include <map>
#include <time.h>

// libsvm
#include "SVM.h"
 

//opencv libraries
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/face/facerec.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

struct Facetest{
	Mat face_t;
	string name_t;
    int labelValue_t;
};

struct Facetrain{
    Mat face;
    string name;
    int labelValue;
};

struct Interval_train{
    vector<Facetrain>::iterator start_train; 
    vector<Facetrain>::iterator end_train;//end is NOT in the INTERVAL!
    Interval_train(vector<Facetrain>::iterator s, vector<Facetrain>::iterator e){//make ref
        start_train = s; 
        end_train = e;
    }
    int Length() const{
        cout<<end_train - start_train<<endl;
        return end_train - start_train;
    }
};

struct Interval_test{
	vector<Facetest>::iterator start_test;
	vector<Facetest>::iterator end_test;
	Interval_test(vector<Facetest>::iterator s_t, vector<Facetest>::iterator e_t){
		start_test = s_t;
		end_test = e_t;
	}
	int Length_t() const{
		cout<<end_test - start_test<<endl;
		return end_test - start_test;
	}

};

class PCASVM{
public:
    //pca parameters
    Mat w,ev,mean;
    vector<Facetrain> projections;
    vector<Facetest> projections_t;
    unsigned int data_type;
    unsigned int width,width_t;
    unsigned int height,height_t;
    unsigned int n,n_t;

    //svm parameters
    svm_parameter param;     // set by parse_command_line
    svm_problem prob;        // set by read_problem
    svm_model *model;
    svm_node *x_space;

    PCASVM(unsigned int data_type_ = CV_32F);
    ~PCASVM(void);

    unsigned int getWidth() const {return width;}
    unsigned int getHeight() const {return height;}
    int train(const list<Interval_test> &intervals_t,const list<Interval_train> &intervals,const bool console_output = true);
};
#endif
