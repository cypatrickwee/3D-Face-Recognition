#include "PCAKNN.h"
#include <iostream>


using namespace std;

PCAKNN::PCAKNN(unsigned int data_type_){
	this->data_type = data_type_;
	//initialization set n, width, height to be 0
	n = n_t = width = height = width_t = height_t = 0;
}

PCAKNN::~PCAKNN(void){}

void PCAKNN::train(const list<Interval_train> &intervals,const bool console_output){
	if(intervals.size() == 0) 
		return;
	projections.clear(); 
	n = 0;

	width = intervals.front().start_train->face.cols;
	cout<<"This is the width"<<endl;
	cout<<width<<endl;
	height = intervals.front().start_train->face.rows;
	cout<<"This is the height"<<endl;
	cout<<height<<endl;

	list<Interval_train>::const_iterator itr;
	vector<Facetrain>::const_iterator start_train,end_train;
	
	//add the number of images from all intervals
	for(itr = intervals.begin(); itr != intervals.end();itr++){
		n += itr->Length();
	}
	

	Mat pca_matrix(static_cast<int>(n), width*height, data_type);
	int c = 0;
	for(itr = intervals.begin();itr != intervals.end();itr++){
		//for each image in the current interval
		for(start_train = itr->start_train,end_train = itr->end_train;start_train != end_train; start_train++,++c){
			if(console_output) printf("Preparing samples %d/%d\n",c+1,n);
			//insert current image into pca_matrix
			Mat image_row = start_train->face.clone().reshape(1,1);
			Mat row_i = pca_matrix.row(c);
			image_row.convertTo(row_i,data_type);//CV_64FC1 ?
			Facetrain f; 
			f.name = start_train->name;
			projections.push_back(f);//save the names for later
		}
	}

	if(console_output) printf("TRAINING...\n");
	//Perfrom principal component analysis on pca_matrix
	PCA pca(pca_matrix,Mat(), CV_PCA_DATA_AS_ROW,pca_matrix.rows);

	//extract mean/eigenvalues
	mean = pca.mean.reshape(1,1);
	ev = pca.eigenvalues.clone();
	transpose(pca.eigenvectors, w);

	//project each face into subspace and save them with the name above for recognition
	for(unsigned int i = 0; i<n;++i){		
		if(console_output) printf("Projecting %d/%d\n",i+1,n);//project so subspace
		projections[i].face = LDA::subspaceProject(w,mean,pca_matrix.row(i));
	}
	//cout<<projections.face.data()<<endl;
}

//k should be uneven to break ties
string PCAKNN::recognize(const list<Interval_test> &intervals_t, const unsigned int nClass,const bool use_distance_weighting, const bool console_output){
	

	projections_t.clear();

	width_t = intervals_t.front().start_test->face_t.cols;
    cout<<"This is the width of the first testing face: "<<width_t<<endl;
    height_t = intervals_t.front().start_test->face_t.rows;
    cout<<"This is the height of the first testing face: "<<height_t<<endl;

    //iteration declaration for testing
    list<Interval_test>::const_iterator itr_t;
	vector<Facetest>::const_iterator start_test, end_test;

	//add the number of images from all intervals (testing)
    //get the total number of the array list
    for(itr_t = intervals_t.begin(); itr_t != intervals_t.end(); itr_t++){
        n_t = n_t + itr_t->Length_t();
    }
    Mat test_matrix(static_cast<int>(n_t),width_t*height_t, data_type);
    cout<<"until here no problem"<<endl;
	
	int c_t = 0;
    for(itr_t = intervals_t.begin();itr_t != intervals_t.end();itr_t++){
        //for each image in the current interval
        for(start_test = itr_t->start_test,end_test = itr_t->end_test;start_test != end_test; start_test++,++c_t){
            if(console_output) printf("Preparing samples %d/%d\n",c_t+1,n_t);
            //insert current image into pca_matrix
            Mat facet_row = start_test->face_t.clone().reshape(1,1);
            //initialize the row_i matrix to be a format of pca_matrix
            //notice ** pca_matrix.row(c) contains a row matrix with all zeroes
            Mat rowt_i = test_matrix.row(c_t);
            //put the value in image_row to row_i
            facet_row.convertTo(rowt_i,data_type);//CV_64FC1 ?
            Facetest ft; 
            ft.name_t = start_test->name_t;
            projections_t.push_back(ft);//save the names for later
        }
    }
    int correct_count =0; 
    int wrong_count =0;
    string name = "N/A";
	if(n <= nClass) return name;
    for(unsigned int fstCount=0;fstCount<n_t;++fstCount){
    	if(console_output){
            printf("Projecting %d/%d\n",fstCount+1,n_t);//project to subspace (testing)
        }
		//project target face to subspace
        projections_t[fstCount].face_t = LDA::subspaceProject(w,mean,test_matrix.row(fstCount));
	
		vector<unsigned int> classes(nClass,0);
		vector<double> distances(nClass,DBL_MAX);
		double dist = DBL_MAX;
		//find k nearest neighbours
		for(unsigned int sndCount = 0; sndCount < projections.size();++sndCount){
			dist = norm(projections[sndCount].face,projections_t[fstCount].face_t,NORM_L2);//norml2
			for(unsigned int trdCount = 0; trdCount < nClass;++trdCount){
				if(dist < distances[trdCount]){
					//discard the worst match and shift remaining down 
					for(int fthCount = nClass-1; fthCount > trdCount;--fthCount){
						distances[fthCount] = distances[fthCount-1];
						classes[fthCount] = classes[fthCount-1];
					}
					classes[trdCount] = sndCount; //set new best match
					distances[trdCount] = dist;
					break;
				}
			}
		}
		map<string,Weighting> neighbours;
		//count occurence of classes
		for(unsigned int i = 0; i<nClass;++i){
			Weighting &weight = neighbours[projections[classes[i]].name];
			weight.count++;
			weight.total_dist += distances[i];
		}
		//evaluate voting
		if(use_distance_weighting){
			double min_weight = DBL_MAX;
			for (map<string,Weighting>::iterator itr = neighbours.begin(); itr != neighbours.end();++itr){
				double weight = itr->second.total_dist / (double) itr->second.count;
				//concider average weight instead of number of votes
				if(weight < min_weight){
					min_weight = weight;
					name = itr->first;
			} 
			if(console_output)
				printf("Weighting points: %s %f\n",itr->first.c_str(),weight);
			}
		}
		else{
			unsigned int max_count = 0;
			//choose the class with the most votes
			for (map<string,Weighting>::iterator itr = neighbours.begin(); itr != neighbours.end();++itr){
				if(itr->second.count > max_count){
					max_count = itr->second.count;
					name = itr->first;
				} 
				if(console_output)
				printf("Voting points: %s %d\n",itr->first.c_str(),itr->second.count);
			}
		}		    	

		cout<<"The tested faces "<<fstCount<<" is: "<<projections_t[fstCount].name_t<<endl;
		if(name==projections_t[fstCount].name_t){
    		++correct_count;
        	cout<<"The output of the recognized class: "<<name<<endl;
    	}
   		else{
        	++wrong_count;
        	cout<<"The output of the recognized class: "<<name<<endl;
    	}


    }
	cout << "done" << endl;
    cout << "RESULT : correct=" << correct_count << " : wrong=" << wrong_count << endl;
    cout << "Accuracy[%]=" << ( static_cast<double>(correct_count)/static_cast<double>(correct_count+wrong_count)*100.0 ) << endl;    	
}