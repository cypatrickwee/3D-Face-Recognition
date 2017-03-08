#include "PCAKNN.h"
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include <time.h>

int main(int argc, const char *argv[]) {

	printf("Usage: dataset.txt target.png k\n",argc);

	if(argc < 4){
		cout<<"LOLOLx Error la okay??"<<endl;
		return EXIT_FAILURE;
	}
    //ifstream for training faces
    ifstream datasetlist(argv[1],ios::in);
    //ifstream for testing faces
    ifstream testinglist(argv[2],ios::in);

    //error checking
    if(testinglist.is_open() == false || datasetlist.is_open() == false){
        printf("Unable to open datasetlist: %s\n or testinglist: %s\n",argv[1],argv[2]);
        return EXIT_FAILURE;
    }

//~~~~~~~~~~~~~~~~~~start to read faces from the file (testing set)~~~~~~~~~~~~~~~~~~~~~~~~~~//
	vector<Facetest> faceVector_t;
    string dataline_t, facefile_t;
    //parse list and load testing faces
    while(testinglist.good() && getline(testinglist,dataline_t)){	
    	//get the current line
    	stringstream ss(dataline_t);
    	//split it at the separator
    	getline(ss,facefile_t,';');
    	//if no faces anymore jump out the loop
    	if(facefile_t.empty()) continue;
    	string label_t; //declare a label variable of type string for the name storage
    	ss >> label_t;
    	//printout the directory path
    	cout<<"Loading: "<<facefile_t.c_str()<<endl;
/*
*	below this is 3D point extraction algorithm
*/
        vtkSmartPointer<vtkPolyDataReader> reader =
        vtkSmartPointer<vtkPolyDataReader>::New();
        //Here you need to know where is the path, we could just use path.c_str to indicate the file
        //note each while loop only access the file once. Hence the path.c_str will only store 1 path
        //stringstream pathess(path);
        reader->SetFileName(facefile_t.c_str());
        reader->Update();
        vtkPolyData* fileData = reader->GetOutput();

        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkPolyData> polydataResult = vtkSmartPointer<vtkPolyData>::New();

        //There are a total of 5090 points for one face
        int nOfPnts = fileData->GetNumberOfPoints();

        //the dimension is (x,y,z)
        int arrayDim = 3;
        
        //a vector length of nOfPnts initialized to vector<float>(arrayDim) of 3 zeros internally
        vector<vector<float> >arrayPts_t(nOfPnts, vector<float>(arrayDim));
       
        //initalize the matrix into a 5090x3 matrix for testing set
        Mat mat_t(nOfPnts, arrayDim, CV_32FC1);
 
        for (vtkIdType i = 0; i < nOfPnts; i++){
                double p[3];
                fileData->GetPoint(i, p);
                for (int j = 0; j <= 2; j++){
                    arrayPts_t[i][j] = p[j];
                    //using this syntax to print the output
                    //note the output will be
                    //-11.5196 55.5718 -1506.76
                    //cout << arrayPts[i][j] << " ";
                    mat_t.at<float>(i,j)=arrayPts_t[i][j];
                }

        }
        //declare an object for variable accessing purposes within the struct facetest struct
        Facetest f_t;
        f_t.face_t = mat_t;
        f_t.name_t = label_t;
        faceVector_t.push_back(f_t);

  	}
  	//close the file reading material
  	testinglist.close();

//~~~~~~~~~~~~~~~~~start to read faces from the file (training set)~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    //parse list
    string dataline,facefile;
    vector<Facetrain> faceVector;
    int count = 0;
    //parse list and load training faces
    while(datasetlist.good() && getline(datasetlist,dataline)){
        //get the current line:
        stringstream ss(dataline);
        //split it at the separator
        getline(ss,facefile,';');
        //if no imagefile jump out the loop
        if(facefile.empty()) continue;
        string label;
        //read the name from the stringstream object ss to label
        ss >> label;
        //printout the pathway      
        printf("Loading %s\n",facefile.c_str());


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Point Extraction~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
 
        vtkSmartPointer<vtkPolyDataReader> reader =
        vtkSmartPointer<vtkPolyDataReader>::New();
        //Here you need to know where is the path, we could just use path.c_str to indicate the file
        //note each while loop only access the file once. Hence the path.c_str will only store 1 path
        //stringstream pathess(path);
        reader->SetFileName(facefile.c_str());
        reader->Update();
        vtkPolyData* fileData = reader->GetOutput();

        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkPolyData> polydataResult = vtkSmartPointer<vtkPolyData>::New();

        //There are a total of 5090 points for one face
        int nOfPnts = fileData->GetNumberOfPoints();

        //the dimension is (x,y,z)
        int arrayDim = 3;
        //a vector length of nOfPnts initialized to vector<float>(arrayDim) of 3 zeros internally
        vector<vector<float> >arrayPts(nOfPnts, vector<float>(arrayDim));
        //initalize the matrix into a 5090x3 matrix
        Mat mat(nOfPnts, arrayDim, CV_32FC1);
        //getchar();
        //loop through every points
        for (vtkIdType i = 0; i < nOfPnts; i++){
                double p[3];
                fileData->GetPoint(i, p);

                for (int j = 0; j <= 2; j++){
                    arrayPts[i][j] = p[j];
                    //using this syntax to print the output
                    //note the output will be
                    //-11.5196 55.5718 -1506.76
                    //cout << arrayPts[i][j] << " ";
                    mat.at<float>(i,j)=arrayPts[i][j];
                }

        }
        //note: the entire vector format is behave something like vec{(vec(0),vec(1),vec(2))...}
        //cout << endl << vec.size() << endl;//output 5090, the reason is because each vec contains 3 points
        cout << "This is the complete number of points of no." << count << " face." << endl;
        count++;
        //getchar();

        Facetrain f; 
        f.name = label; 
        f.face = mat;
        faceVector.push_back(f);     
	}

	datasetlist.close();

	if(faceVector_t.size() < 1 || faceVector.size() < 1){
        printf("Loading face error.\n");
        return EXIT_FAILURE;
    }


	try{
		//PCA training
		PCAKNN pcaknn;
        //list of training faces
        list<Interval_train> intervals;
        //list of testing faces
        list<Interval_test> intervals_t;
        //training faces interval
        intervals.push_back(Interval_train(faceVector.begin(),faceVector.end()));
        //testing faces interval
        intervals_t.push_back(Interval_test(faceVector_t.begin(),faceVector_t.end()));

        //run pcaknn training (train the model)
		pcaknn.train(intervals);

		//Size dim = Size(pcaknn.getWidth(), pcaknn.getHeight());

			//load file from comandline
			//string inputfile(argv[2]);
			unsigned int k = (unsigned int)atoi(argv[3]);
			//Mat target = imread(inputfile,0);

		//run KNN
		pcaknn.recognize(intervals_t,k,k % 2 == 0);	

	}catch(const cv::Exception &e){
		printf("%s\n",e.what());
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}