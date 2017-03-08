#include "PCASVM.h"

//PCASVM class initialization
PCASVM::PCASVM(unsigned int data_type_){
    this->data_type = data_type_;
    //initialization set n, width, height to be 0
    n = n_t = width = height = width_t = height_t = 0;
}
PCASVM::~PCASVM(void){}


int PCASVM::train(const list<Interval_test> &intervals_t, const list<Interval_train> &intervals,const bool console_output){
    if(intervals.size() == 0) 
        return EXIT_FAILURE;
    
    projections.clear();
    projections_t.clear();
    n = 0;
    n_t = 0;

    width = intervals.front().start_train->face.cols;
    cout<<"This is the width of the first training face: "<<width<<endl;
    height = intervals.front().start_train->face.rows;
    cout<<"This is the height of the first training face: "<<height<<endl;
    width_t = intervals_t.front().start_test->face_t.cols;
    cout<<"This is the width of the first testing face: "<<width_t<<endl;
    height_t = intervals_t.front().start_test->face_t.rows;
    cout<<"This is the height of the first testing face: "<<height_t<<endl;

    
    //iteration declaration for training
    list<Interval_train>::const_iterator itr;
    vector<Facetrain>::const_iterator start_train, end_train;

    //iteration declaration for testing
    list<Interval_test>::const_iterator itr_t;
	vector<Facetest>::const_iterator start_test, end_test;
    
    //add the number of images from all intervals (training)
    //get the total number of the array list
    for(itr = intervals.begin(); itr != intervals.end();itr++){
        n += itr->Length();
    }

    //add the number of images from all intervals (testing)
    //get the total number of the array list
    for(itr_t = intervals_t.begin(); itr_t != intervals_t.end(); itr_t++){
        n_t = n_t + itr_t->Length_t();
    }
    //declare a matrix with mat type and has the row as n, column as width*height
    //with data type as data_type
    //notice ** the interval values are all zeroes
    Mat train_matrix(static_cast<int>(n), width*height, data_type);
    Mat test_matrix(static_cast<int>(n_t),width_t*height_t, data_type);
    
    
    //storing the training faces into a matrix name row_i
    //subsequently stored into a vector with name projections
    //different between ++a and a++; ++a add 1 then only do everything
    int c = 0;
    int c_t = 0;
    //after this for loop "train_matrix" will contains the entire value of the faces points
    //of all faces from the training data
    for(itr = intervals.begin();itr != intervals.end();itr++){
        //for each image in the current interval
        for(start_train = itr->start_train, end_train = itr->end_train;start_train != end_train; start_train++,++c){
            if(console_output) printf("Preparing samplesss %d/%d\n",c+1,n);
            //insert current image into pca_matrix
            Mat face_row = start_train->face.clone().reshape(1,1);
            //initialize the row_i matrix to be a format of pca_matrix
            //notice ** pca_matrix.row(c) contains a row matrix with all zeroes
            Mat row_i = train_matrix.row(c);
            //put the value in image_row to row_i
            face_row.convertTo(row_i,data_type);//CV_64FC1 ?
            Facetrain f; 
            f.name = start_train->name;
            projections.push_back(f);//save the names for later
        }
    }
    //same as previous for loop this is for testing faces
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


    if(console_output) printf("TRAINING...\n");
    //Perfrom principal component analysis on pca_matrix
    //PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, num_components);
    PCA pca(train_matrix,Mat(), CV_PCA_DATA_AS_ROW,train_matrix.rows);

    //extract mean/eigenvalues
    mean = pca.mean.reshape(1,1);
    ev = pca.eigenvalues.clone();//eigenvalues deep copy to variable ev
    transpose(pca.eigenvectors, w);//transpose the pca.eigenvectors and store in w

    //project each face into subspace and save them with the name above for recognition
    //this is actually the model training, something like the stuff did in tutSVM train section
    for(unsigned int i = 0; i<n;++i){       
        if(console_output){
            printf("Projecting %d/%d\n",i+1,n);//project so subspace
        }
        projections[i].face = LDA::subspaceProject(w,mean,train_matrix.row(i));
        //cout<<projections[i].face<<endl;
        cout<<"This is the rows: "<<projections[i].face.rows<<endl;
        cout<<"This is the cols: "<<projections[i].face.cols<<endl;
        cout<<"This is the name with respect to each row/face: "<<projections[i].name<<endl;
        if (projections[i].name=="mdm100"){
            projections[i].labelValue = 1;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="gp200e"){
            projections[i].labelValue = 2;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="om00e"){
            projections[i].labelValue = 3;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="wyt98"){
            projections[i].labelValue = 4;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="ic100e"){
            projections[i].labelValue = 5;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="drj00"){
            projections[i].labelValue = 6;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="jr200"){
            projections[i].labelValue = 7;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="dsl00m"){
            projections[i].labelValue = 8;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="pjc99"){
            projections[i].labelValue = 9;
            cout<<"label done for "<<projections[i].name<<"."<<endl; 
        }
        else if(projections[i].name=="tcl00"){
            projections[i].labelValue = 10;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="mf200"){
            projections[i].labelValue = 11;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="gatj98"){
            projections[i].labelValue = 12;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="jnh00"){
            projections[i].labelValue = 13;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="arr00"){
            projections[i].labelValue = 14;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="crn99"){
            projections[i].labelValue = 15;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="rb200m"){
            projections[i].labelValue = 16;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="bdr00"){
            projections[i].labelValue = 17;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="jac100e"){
            projections[i].labelValue = 18;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="dg100e"){
            projections[i].labelValue = 19;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="jl400"){
            projections[i].labelValue = 20;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="sc300m"){
            projections[i].labelValue = 21;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="jasa98"){
            projections[i].labelValue = 22;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="mha00"){
            projections[i].labelValue = 23;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="tt100"){
            projections[i].labelValue = 24;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="cjf00"){
            projections[i].labelValue = 25;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="svk00"){
            projections[i].labelValue = 26;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="pjw00"){
            projections[i].labelValue = 27;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="tjw00m"){
            projections[i].labelValue = 28;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="xw00"){
            projections[i].labelValue = 29;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="pm00"){
            projections[i].labelValue = 30;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="jb300e"){
            projections[i].labelValue = 31;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="dm100"){
            projections[i].labelValue = 32;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="pba99"){
            projections[i].labelValue = 33;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="jcl100m"){
            projections[i].labelValue = 34;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="mzm00"){
            projections[i].labelValue = 35;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="dwr00"){
            projections[i].labelValue = 36;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="ad499"){
            projections[i].labelValue = 37;
            cout<<"label done for "<<projections[i].name<<"."<<endl; 
        }
        else if(projections[i].name=="jdd00e"){
            projections[i].labelValue = 38;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="mh500"){
            projections[i].labelValue = 39;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="vvw00e"){
            projections[i].labelValue = 40;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="dg00"){
            projections[i].labelValue = 41;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="ks200"){
            projections[i].labelValue = 42;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="sft99"){
            projections[i].labelValue = 43;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="kl100"){
            projections[i].labelValue = 44;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="tjh99"){
            projections[i].labelValue = 45;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="yjs00e"){
            projections[i].labelValue = 46;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="hsn00e"){
            projections[i].labelValue = 47;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="cmt00m"){
            projections[i].labelValue = 48;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="era02"){
            projections[i].labelValue = 49;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="ap400"){
            projections[i].labelValue = 50;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="lm02"){
            projections[i].labelValue = 51;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="jdl99"){
            projections[i].labelValue = 52;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="paj00"){
            projections[i].labelValue = 53;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="tel99"){
            projections[i].labelValue = 54;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="jwa00"){
            projections[i].labelValue = 55;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="chy00"){
            projections[i].labelValue = 56;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="jcs100"){
            projections[i].labelValue = 57;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="jhb02"){
            projections[i].labelValue = 58;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="pmn00e"){
            projections[i].labelValue = 59;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else if(projections[i].name=="mp400"){
            projections[i].labelValue = 60;
            cout<<"label done for "<<projections[i].name<<"."<<endl;
        }
        else{
            cout<<"There is an error here."<<endl;
        }
        //getchar();

        //IMPORTANT
        //so far as I know for SVM, you need to train out a model
        //e.g. svm_model* model = svm_train( &prob, &param );
        //next, using the test model to predict the outcome
        //const auto kPredictedLabel = static_cast<int>( svm_predict( model, test ) );
        //things to take notes over here:
        //i. &prob and test will have the format of libsvm readout format
        //ii. you can actually output the training set eigenvectors that has been projected to the subspace
        //into a file with that format and so goes to test set as well.
    }
    //build the SVM training model
    cout<<"At the same time building the SVM model:~~~"<<endl;
//~~~~~~~~~~~~~~~~~~~~~~~~~~define the svm problem~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    svm_problem prob;
    //number of training data
    prob.l = projections.size();
    cout<<prob.l<<endl;

    //initialize an array
    prob.y = new double[ prob.l ];
    int count=0;
    for(size_t i=0; i<projections.size(); ++i ){
        prob.y[i] = projections[i].labelValue;
        cout<<"This is prob.y[i]: "<<prob.y[i]<<endl;
        count++;
    }
    cout<<"Total number of labeled data: "<<count<<endl;
    int columns = projections[prob.l-1].face.cols;

//~~~~~~~~~~~~~~~~~~~~~~create the SVM model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //svm_node *prob_vec = new svm_node[prob.l*(1+columns)];
    prob.x = new svm_node*[prob.l];//e.g. int *array = new int*[10];
    svm_node** x = new svm_node*[prob.l];

    for (int row = 0;row <prob.l; row++){
        svm_node* x_space = new svm_node[columns+1];
        for (int col = 0;col < columns;col++){
            x_space[col].index = col;
            x_space[col].value = projections[row].face.at<float>(col);
        }
        x_space[columns].index = -1;      //Each row of properties should be terminated with a -1 according to the readme
        x[row] = x_space;
    }

    prob.x = x;

//~~~~~~~~~~~~~~~~~~~~~~define the parameters of SVM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    svm_parameter param;
    param.svm_type = NU_SVC;//multi-label classifier type
    param.kernel_type = LINEAR;//polynomial kernel
    param.C = 10000;
    param.gamma = 0.1;
 

    param.coef0 = 0;//try and error
    param.cache_size = 1000000;
    param.eps = 1e-3;
    param.shrinking = 0;
    param.probability = 0;
 
    param.degree = 3;
    param.nu = 0.5;
    param.p = 0.1;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Train model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    svm_model *model = svm_train(&prob,&param);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Recognition~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int correct_count =0; 
    int wrong_count =0;
    for(unsigned int j=0;j<n_t;++j){
        if(console_output){
            printf("Projecting %d/%d\n",j+1,n_t);//project so subspace
        }
        projections_t[j].face_t = LDA::subspaceProject(w,mean,test_matrix.row(j));
        svm_node* test = new svm_node[projections_t[j].face_t.cols];
        for (int i=0; i<projections_t[j].face_t.cols;i++){
            test[i].index = i;
            test[i].value = (double)projections_t[j].face_t.at<float>(i);
        }
        test[projections_t[j].face_t.cols].index = -1;
        double retval = svm_predict(model,test);
        cout<<"The output of the predicted class: "<<retval<<endl;
        if (projections_t[j].name_t=="mdm100"){
            projections_t[j].labelValue_t = 1;
        }
        else if(projections_t[j].name_t=="gp200e"){
            projections_t[j].labelValue_t = 2;
        }
        else if(projections_t[j].name_t=="om00e"){
            projections_t[j].labelValue_t = 3;
        }
        else if(projections_t[j].name_t=="wyt98"){
            projections_t[j].labelValue_t = 4;
        }
        else if(projections_t[j].name_t=="ic100e"){
            projections_t[j].labelValue_t = 5;
        }
        else if(projections_t[j].name_t=="drj00"){
            projections_t[j].labelValue_t = 6;
        }
        else if(projections_t[j].name_t=="jr200"){
            projections_t[j].labelValue_t = 7;
        }
        else if(projections_t[j].name_t=="dsl00m"){
            projections_t[j].labelValue_t = 8;
        }
        else if(projections_t[j].name_t=="pjc99"){
            projections_t[j].labelValue_t = 9;
        }
        else if(projections_t[j].name_t=="tcl00"){
            projections_t[j].labelValue_t = 10;
        }
        else if(projections_t[j].name_t=="mf200"){
            projections_t[j].labelValue_t = 11;
        }
        else if(projections_t[j].name_t=="gatj98"){
            projections_t[j].labelValue_t = 12;
        }
        else if(projections_t[j].name_t=="jnh00"){
            projections_t[j].labelValue_t = 13;
        }
        else if(projections_t[j].name_t=="arr00"){
            projections_t[j].labelValue_t = 14;
        }
        else if(projections_t[j].name_t=="crn99"){
            projections_t[j].labelValue_t = 15;
        }
        else if(projections_t[j].name_t=="rb200m"){
            projections_t[j].labelValue_t = 16;
        }
        else if(projections_t[j].name_t=="bdr00"){
            projections_t[j].labelValue_t = 17;
        }
        else if(projections_t[j].name_t=="jac100e"){
            projections_t[j].labelValue_t = 18;
        }
        else if(projections_t[j].name_t=="dg100e"){
            projections_t[j].labelValue_t = 19;
        }
        else if(projections_t[j].name_t=="jl400"){
            projections_t[j].labelValue_t = 20;
        }
        else if(projections_t[j].name_t=="sc300m"){
            projections_t[j].labelValue_t = 21;
        }
        else if(projections_t[j].name_t=="jasa98"){
            projections_t[j].labelValue_t = 22;
        }
        else if(projections_t[j].name_t=="mha00"){
            projections_t[j].labelValue_t = 23;
        }
        else if(projections_t[j].name_t=="tt100"){
            projections_t[j].labelValue_t = 24;
        }
        else if(projections_t[j].name_t=="cjf00"){
            projections_t[j].labelValue_t = 25;
        }
        else if(projections_t[j].name_t=="svk00"){
            projections_t[j].labelValue_t = 26;
        }
        else if(projections_t[j].name_t=="pjw00"){
            projections_t[j].labelValue_t = 27;
        }
        else if(projections_t[j].name_t=="tjw00m"){
            projections_t[j].labelValue_t = 28;
        }
        else if(projections_t[j].name_t=="xw00"){
            projections_t[j].labelValue_t = 29;
        }
        else if(projections_t[j].name_t=="pm00"){
            projections_t[j].labelValue_t = 30;
        }
        else if(projections_t[j].name_t=="jb300e"){
            projections_t[j].labelValue_t = 31;
        }
        else if(projections_t[j].name_t=="dm100"){
            projections_t[j].labelValue_t = 32;
        }
        else if(projections_t[j].name_t=="pba99"){
            projections_t[j].labelValue_t = 33;
        }
        else if(projections_t[j].name_t=="jcl100m"){
            projections_t[j].labelValue_t = 34;
        }
        else if(projections_t[j].name_t=="mzm00"){
            projections_t[j].labelValue_t = 35;
        }
        else if(projections_t[j].name_t=="dwr00"){
            projections_t[j].labelValue_t = 36;
        }
        else if(projections_t[j].name_t=="ad499"){
            projections_t[j].labelValue_t = 37; 
        }
        else if(projections_t[j].name_t=="jdd00e"){
            projections_t[j].labelValue_t = 38;
        }
        else if(projections_t[j].name_t=="mh500"){
            projections_t[j].labelValue_t = 39;
        }
        else if(projections_t[j].name_t=="vvw00e"){
            projections_t[j].labelValue_t = 40;
        }
        else if(projections_t[j].name_t=="dg00"){
            projections_t[j].labelValue_t = 41;
        }
        else if(projections_t[j].name_t=="ks200"){
            projections_t[j].labelValue_t = 42;
        }
        else if(projections_t[j].name_t=="sft99"){
            projections_t[j].labelValue_t = 43;
        }
        else if(projections_t[j].name_t=="kl100"){
            projections_t[j].labelValue_t = 44;
        }
        else if(projections_t[j].name_t=="tjh99"){
            projections_t[j].labelValue_t = 45;
        }
        else if(projections_t[j].name_t=="yjs00e"){
            projections_t[j].labelValue_t = 46;
        }
        else if(projections_t[j].name_t=="hsn00e"){
            projections_t[j].labelValue_t = 47;
        }
        else if(projections_t[j].name_t=="cmt00m"){
            projections_t[j].labelValue_t = 48;
        }
        else if(projections_t[j].name_t=="era02"){
            projections_t[j].labelValue_t = 49;
        }
        else if(projections_t[j].name_t=="ap400"){
            projections_t[j].labelValue_t = 50;
        }
        else if(projections_t[j].name_t=="lm02"){
            projections_t[j].labelValue_t = 51;
        }
        else if(projections_t[j].name_t=="jdl99"){
            projections_t[j].labelValue_t = 52;
        }
        else if(projections_t[j].name_t=="paj00"){
            projections_t[j].labelValue_t = 53;
        }
        else if(projections_t[j].name_t=="tel99"){
            projections_t[j].labelValue_t = 54;
        }
        else if(projections_t[j].name_t=="jwa00"){
            projections_t[j].labelValue_t = 55;
        }
        else if(projections_t[j].name_t=="chy00"){
            projections_t[j].labelValue_t = 56;
        }
        else if(projections_t[j].name_t=="jcs100"){
            projections_t[j].labelValue_t = 57;
        }
        else if(projections_t[j].name_t=="jhb02"){
            projections_t[j].labelValue_t = 58;
        }
        else if(projections_t[j].name_t=="pmn00e"){
            projections_t[j].labelValue_t = 59;
        }
        else if(projections_t[j].name_t=="mp400"){
            projections_t[j].labelValue_t = 60;
        }
        else{
            cout<<"There is an error here."<<endl;
        }
        if (retval==projections_t[j].labelValue_t)
        {
            ++correct_count;
            cout<<"The output of the recognized class: "<<projections_t[j].labelValue_t<<endl;
        }
        else
        {
            ++wrong_count;
            cout<<"The output of the recognized class: "<<projections_t[j].labelValue_t<<endl;
        }
    }
    cout << "done" << endl;
    cout << "RESULT : correct=" << correct_count << " : wrong=" << wrong_count << endl;
    cout << "Accuracy[%]=" << ( static_cast<double>(correct_count)/static_cast<double>(correct_count+wrong_count)*100.0 ) << endl;       
    svm_free_and_destroy_model( &model );
    delete[] prob.y;
    delete[] prob.x;
}