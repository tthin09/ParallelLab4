#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <array>
#include <thread>
#include <queue>
#include <mutex>

// Constant declaration
const int MAP_SIZE = 4000;
const int ITERATIONS = 5;
const float HEAT_KERNEL[3][3] = { {0.05, 0.1, 0.05},
                                {0.1, 0.4, 0.1},
                                {0.05, 0.1, 0.05} };
const float env_temp = 30;
const float env_radi = 0; 
const int D = 1000;
const double k = 0.0001;
const double lambda = 0.00003;
const double ux = 3.3;
const double uy = 1.4;
const int Deg = 8;
const std::array<double, 9> Coeffs = {2.611369, -1.690128, 0.00805, 0.336743, -0.005162, -0.080923, -0.004785, 0.007930, 0.000768};
const int spread_speed = 343;
const float EXPLODE_POS = MAP_SIZE*0.5-0.5;
const int WEIGHT = 5*pow(10,6);
const int INIT = 0;
const int NUM_THREAD = 4;
const std::string LAB1 = "Heat_Diffusion.csv";
const std::string LAB2 = "Radioactive_Diffusion.csv";
const std::string LAB3 = "Shock_wave_blast.csv";

// Functions
double distance(double x_1, double y_1, double x_2, double y_2){
    return pow(pow(abs(x_2-x_1)/10, 2) + pow(abs(y_2-y_1)/10, 2),0.5)*10;
}

double scale_Distance(double distance, double explosive_yield){
    return distance*std::pow(explosive_yield,-1/3);
}

double peak_Overpressure(double scale_distance){
    if(scale_distance==0) return INFINITY;
    double u = -0.21436 + 1.35034*std::log10(scale_distance);
    double temp = 1;
    double sum = 0;
    for(int i = 0;i<=Deg;i++){
        sum+=temp*Coeffs[i];
        temp*=u;
    }
    return pow(10,sum);
}

void getDataFromCsv(double** array, const char* filename){
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }
    int i = 0;
    int j = 0;
    std::string line;
    while (std::getline(file, line)){
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')){
            array[i][j]=std::stoi(cell);
            j++;
        }
        j=0;
        i++;
    }
    file.close();
    return;
}

// Task queue structure
struct Task{
    int id_x,id_y;
    Task(int id_x,int id_y): id_x(id_x),id_y(id_y){}
};
class TaskQueue{
    private:
        std::mutex mtx;
        std::queue<Task*> queue;
        static TaskQueue* instance;
    
    public:
        void enqueue(Task*task){
            std::lock_guard<std::mutex> lock(mtx);
            queue.push(task);
        }

        Task*dequeue(){
            std::lock_guard<std::mutex> lock(mtx);
            if(!queue.empty()) {
                Task*task= queue.front();
                queue.pop();
                return task;
            }
            return nullptr;
        }

        bool empty(){
            return queue.empty();
        }

        static TaskQueue*get(){
            if(instance== nullptr)
                instance=new TaskQueue();
            return instance;
        }
};

class Map{
    private:
        int size;
        double** map;
        std::mutex map_mtx;
    public:
        Map(int size, double init, double** map): size(size), map(map){}
        double get(int x, int y){
            std::lock_guard<std::mutex> lock(map_mtx);
            return map[x][y];
        }
        void set(int x, int y, double val){
            std::lock_guard<std::mutex> lock(map_mtx);
            map[x][y]=val;
            return;
        }
        void delete_map(){
            // Free memory
            for (int i = 0;i<MAP_SIZE;i++){
                free(map[i]);
                map[i]=nullptr;
            }
            free(map);
            map=nullptr;
        }
};

class Worker{
    private:
        bool stop;
        std::thread t;
        int id;
        Map* map;

        void run(){
            while(!stop){
                Task*task= TaskQueue::get()->dequeue();
                if(task!= nullptr){
                    double dis = distance(task->id_x,task->id_y,EXPLODE_POS,EXPLODE_POS);
                    if(map->get(task->id_x,task->id_y)!=0){
                        std::cout<<"ERROR "<<task->id_x<<" "<<task->id_y<<" "<<dis<<std::endl;
                    }
                    map->set(task->id_x,task->id_y,peak_Overpressure(scale_Distance(dis,WEIGHT)));
                    
                    // //Sleep briefly to avoid busy-waiting(optional)
                    // std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    delete task;
                }else{
                    //Sleep briefly to avoid busy-waiting(optional)
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
        }

    public:
        Worker(int id, Map* map): id(id),stop(false),map(map){
            t=std::thread(&Worker::run,this);
        }
        
        void exit(){
            stop=true;
            t.join();
        }
};
TaskQueue*TaskQueue::instance=nullptr;

// Do 1 iter of Lab3: Shock Wave Blast at n seconds after the blast
void shockWaveBlast(Map* map, double& affected_radius, int& affected_area){
    affected_radius+=3.43;
    if(MAP_SIZE%2==0){
        for(int i=0;i<=affected_radius*10;i++){
            if(i>=MAP_SIZE/2){
                break;
            }
            for(int j=0;j<=affected_radius*10;j++){
                if(j>=MAP_SIZE/2){
                    break;
                }
                if(map->get(MAP_SIZE/2+i,MAP_SIZE/2+j)!=0){
                    continue;
                }
                double dis = distance(MAP_SIZE*0.5+i,MAP_SIZE*0.5+j,EXPLODE_POS,EXPLODE_POS);
                if(dis>affected_radius*10){
                    break;
                }else{
                    TaskQueue::get()->enqueue(new Task(MAP_SIZE/2+i,MAP_SIZE/2+j));
                    TaskQueue::get()->enqueue(new Task(MAP_SIZE/2-1-i,MAP_SIZE/2+j));
                    TaskQueue::get()->enqueue(new Task(MAP_SIZE/2+i,MAP_SIZE/2-1-j));
                    TaskQueue::get()->enqueue(new Task(MAP_SIZE/2-1-i,MAP_SIZE/2-1-j));
                    affected_area+=4;
                }
            }
        }
    }else{
        int center = MAP_SIZE/2;
        for(int i=0;i<=affected_radius*10;i++){
            if(i>(MAP_SIZE-1)/2){
                break;
            }
            for(int j=0;j<=affected_radius*10;j++){
                if(j>(MAP_SIZE-1)/2){
                    break;
                }
                if(map->get(center+i,center+j)!=0){
                    continue;
                }
                double dis = distance(center+i,center+j,EXPLODE_POS,EXPLODE_POS);
                if(dis>affected_radius*10){
                    break;
                }else{
                    TaskQueue::get()->enqueue(new Task(center+i,center+j));
                    TaskQueue::get()->enqueue(new Task(center-i,center+j));
                    TaskQueue::get()->enqueue(new Task(center+i,center-j));
                    TaskQueue::get()->enqueue(new Task(center-i,center-j));
                    if(i==0&&j==0){
                        affected_area+=1;
                    }else if(i==0||j==0){
                        affected_area+=2;
                    }else{
                        affected_area+=4;
                    }
                }
            }
        }
    }
    //Allow workers sometime to process tasks
    while(!TaskQueue::get()->empty()){
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

// Do 1 iter of Lab 1 : Heat diffusion
void heatDiffusion(double** map, int size_x, int size_y, double* left_pad, double* right_pad, double & peak_temp){
    // Create copy matrix with padding
    double** temp = (double**) malloc(sizeof(double*)*(size_y+2));
    for(int i=0;i<size_y+2;i++){
        temp[i]=(double*) malloc(sizeof(double)*(size_x+2));
        for(int j=0;j<size_x+2;j++){
            if(i==0||i==size_y+1){
                temp[i][j]=0;
            }else if(j==0){
                temp[i][j]=left_pad[i-1];
            }else if(j==size_x+1){
                temp[i][j]=right_pad[i-1];
            }else{
                temp[i][j]=map[i-1][j-1];
            }
        }
    }
    peak_temp=30;
    // Heat diffusion
    #pragma omp parallel for schedule(guided,10000) collapse(2)
    for(int i=1;i<size_y;i++){
        for(int j=1;j<size_x;j++){
            map[i][j]=0;
            for(int h = 0; h<3;h++){
                for(int k = 0;k<3;k++){
                    map[i][j]+=HEAT_KERNEL[h][k]*temp[i+h-1][j+k-1];
                }
            }
            if(map[i][j]>peak_temp){
                peak_temp=map[i][j];
            }
        }
    }
    for(int i=0;i<size_y+2;i++){
        free(temp[i]);
        temp[i]=nullptr;
    }
    free(temp);
    temp=nullptr;
    return;
}

// Do 1 iter of Lab 2 : Radioactive diffusion
void radiDiffusion(double** map,int size_x, int size_y, double* left_pad, double* right_pad, int &cont_num){
    // Create copy matrix with padding
    double** temp = (double**) malloc(sizeof(double*)*(size_y+2));
    for(int i=0;i<size_y+2;i++){
        temp[i]=(double*) malloc(sizeof(double)*(size_x+2));
        for(int j=0;j<size_x+2;j++){
            if(i==0||i==size_y+1){
                temp[i][j]=0;
            }else if(j==0){
                temp[i][j]=left_pad[i-1];
            }else if(j==size_x+1){
                temp[i][j]=right_pad[i-1];
            }else{
                temp[i][j]=map[i-1][j-1];
            }
        }
    }
    cont_num=0;
    // Radioactive diffusion
    #pragma omp parallel for schedule(guided,10000) collapse(2)
    for(int i=1;i<size_y;i++){
        for(int j=1;j<size_x;j++){
            // Wind factor u delta C
            double uDelC = ux*(temp[i+1][j+1]-temp[i][j+1])/10 + uy*(temp[i+1][j+1]-temp[i+1][j])/10;
            // Diffusion factor delta ^2 C
            double del2C = (temp[i+2][j+1]-2*temp[i+1][j+1]+temp[i][j+1])/(10*10) + (temp[i+1][j+2]-2*temp[i+1][j+1]+temp[i+1][j])/(10*10);
            // Update C new
            map[i][j]+= 1*(D*del2C-(lambda+k)*temp[i+1][j+1]-uDelC);
            if(map[i][j]!=0){
                cont_num++;
            }
        }
    }
    for(int i=0;i<size_y+2;i++){
        free(temp[i]);
        temp[i]=nullptr;
    }
    free(temp);
    temp=nullptr;
    return;
}

int main(int argc, char** argv){
    int world_rank=0, world_size=1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Request array for asynchronus implementation
    MPI_Request req[4];
    MPI_Request report_req[2];
    MPI_Request stop_sig;

    if(world_size<3){
        if(world_rank==0){
            printf("Need at least 3 processes\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Start timer
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Only valid on rank=0,1,2, use to get input data
    double** map;
    // Real problem value
    int heat_left, heat_right, heat_top, heat_bottom, radi_left, radi_right, radi_top, radi_bottom;
    heat_left=heat_top=radi_left=radi_top=MAP_SIZE;
    heat_right=heat_bottom=radi_right=radi_bottom=-1;
    map = (double**) malloc(sizeof(double*)*MAP_SIZE);
    // Input processing
    if(world_rank==0){
        // Create data for Lab3
        for (int i = 0;i<MAP_SIZE;i++){
            map[i] = (double*) malloc(sizeof(double)*MAP_SIZE);
            for(int j = 0;j<MAP_SIZE;j++){
                map[i][j]=0;
            }
        }
    }else if(world_rank==1){
        // Get data for Lab1
        for (int i = 0;i<MAP_SIZE;i++){
            map[i] = (double*) malloc(sizeof(double)*MAP_SIZE);
        }
        getDataFromCsv(map,"heat_matrix.csv");
        // Get heat_matrix real problem size using smallest rectangle
        for(int i = 0;i<MAP_SIZE;i++){
            for(int j = 0;j<MAP_SIZE;j++){
                if(map[i][j]!=30){
                    if(j<heat_left){
                        heat_left=j;
                    }
                    if(j>heat_right){
                        heat_right=j;
                    }
                    if(i<heat_top){
                        heat_top = i;
                    }
                    if(i>heat_bottom){
                        heat_bottom=i;
                    }
                }
            }
        }
        heat_left-=ITERATIONS;
        heat_right+=ITERATIONS;
        heat_top-=ITERATIONS;
        heat_bottom+=ITERATIONS;
        int x_size = heat_right-heat_left+1;
        int num_procs = world_size/2;
        int chunk = (x_size+num_procs-1)/num_procs;
        int padding = chunk*num_procs;
        heat_left=heat_left-padding+x_size;
    }else if(world_rank==2){
        // Get data for Lab2
        for (int i = 0;i<MAP_SIZE;i++){
            map[i] = (double*) malloc(sizeof(double)*MAP_SIZE);
        }
        getDataFromCsv(map,"radioactive_matrix.csv");
        // Get radi_matrix real problem size using smallest rectangle
        for(int i = 0;i<MAP_SIZE;i++){
            for(int j = 0;j<MAP_SIZE;j++){
                if(map[i][j]!=0){
                    if(j<radi_left){
                        radi_left=j;
                    }
                    if(j>radi_right){
                        radi_right=j;
                    }
                    if(i<radi_top){
                        radi_top = i;
                    }
                    if(i>radi_bottom){
                        radi_bottom=i;
                    }
                }
            }
        }
        radi_left-=ITERATIONS;
        radi_right+=ITERATIONS;
        radi_top-=ITERATIONS;
        radi_bottom+=ITERATIONS;
        int x_size = radi_right-radi_left+1;
        int num_procs = world_size-world_size/2-1;
        int chunk = (x_size+num_procs-1)/num_procs;
        int padding = chunk*num_procs;
        radi_left=radi_left-padding+x_size;
    }

    // Problem size broadcast
    MPI_Ibcast(&heat_left, 1, MPI_INT, 1, MPI_COMM_WORLD,&req[0]);
    MPI_Ibcast(&heat_right, 1, MPI_INT, 1, MPI_COMM_WORLD,&req[1]);
    MPI_Ibcast(&heat_top, 1, MPI_INT, 1, MPI_COMM_WORLD,&req[2]);
    MPI_Ibcast(&heat_bottom, 1, MPI_INT, 1, MPI_COMM_WORLD,&req[3]);
    MPI_Waitall(4,req,MPI_STATUS_IGNORE);
    MPI_Ibcast(&radi_left, 1, MPI_INT, 2, MPI_COMM_WORLD,&req[0]);
    MPI_Ibcast(&radi_right, 1, MPI_INT, 2, MPI_COMM_WORLD,&req[1]);
    MPI_Ibcast(&radi_top, 1, MPI_INT, 2, MPI_COMM_WORLD,&req[2]);
    MPI_Ibcast(&radi_bottom, 1, MPI_INT, 2, MPI_COMM_WORLD,&req[3]);
    MPI_Waitall(4,req,MPI_STATUS_IGNORE);
    // Real problem size caculation (left, right, top, bottom)
    int left=std::min(heat_left,radi_left);
    int right=std::max(heat_right,radi_right);
    int top=std::min(heat_top,radi_top);
    int bottom=std::max(heat_bottom,radi_bottom);
    int temp_size=right-left+1;
    int temp_num_procs=world_size/2*(world_size-world_size/2-1);
    int temp_chunk=(temp_size+temp_num_procs-1)/temp_num_procs;
    int padding=temp_chunk*temp_num_procs;
    left=left-padding+temp_size;

    // Data preparation
    double** local_h_map; // For odd rank processes (rank = 1,3,5,7,...)
    double** local_r_map; // For even rank processes (rank = 2,4,6,8,...)
    int x_size = right-left+1, y_size=bottom-top+1;
    local_h_map=(double**) malloc(sizeof(double*)*y_size);
    local_r_map=(double**) malloc(sizeof(double*)*y_size);
    int* heat_distribute;
    int* radi_distribute;
    int* heat_offset;
    int* radi_offset;
    heat_distribute=(int*) malloc(sizeof(int)*world_size);
    radi_distribute=(int*) malloc(sizeof(int)*world_size);
    heat_offset=(int*) malloc(sizeof(int)*world_size);
    radi_offset=(int*) malloc(sizeof(int)*world_size);
    int heat_chunk = x_size/(int)(world_size/2);
    int radi_chunk = x_size/(int)(world_size-world_size/2-1);
    if(world_rank%2==0) heat_chunk=0;
    if(world_rank%2==1||world_rank==0) radi_chunk=0;

    for(int i=0;i<world_size;i++){
        if(i==0){
            heat_distribute[i]=0;
            radi_distribute[i]=0;
            heat_offset[i]=left;
            radi_offset[i]=left;
        }else if(i%2==1){
            heat_distribute[i]=heat_chunk;
            radi_distribute[i]=0;
            heat_offset[i]=heat_offset[i-1]+heat_distribute[i-1];
            radi_offset[i]=radi_offset[i-1]+radi_distribute[i-1];
        }else{
            heat_distribute[i]=0;
            radi_distribute[i]=radi_chunk;
            heat_offset[i]=heat_offset[i-1]+heat_distribute[i-1];
            radi_offset[i]=radi_offset[i-1]+radi_distribute[i-1];
        }
    }

    // if(world_rank==3){
    //     for(int i=0;i<world_size;i++){
    //         std::cout<<heat_distribute[i]<<"  "<<heat_offset[i]<<std::endl;
    //     }
    //     for(int i=0;i<world_size;i++){
    //         std::cout<<radi_distribute[i]<<"  "<<radi_offset[i]<<std::endl;
    //     }
    // }
    
    if(world_rank%2==1){
        for(int i=0;i<y_size;i++){
            local_h_map[i]=(double*) malloc(sizeof(double)*heat_chunk);
        }
    }else if(world_rank!=0){
        for(int i=0;i<y_size;i++){
            local_r_map[i]=(double*) malloc(sizeof(double)*radi_chunk);
        }
    }    


    // Scatter data to other processes
    for(int i=0;i<y_size;i++){
        MPI_Scatterv(map[i+top],heat_distribute,heat_offset,MPI_DOUBLE,local_h_map[i],heat_chunk,MPI_DOUBLE,1,MPI_COMM_WORLD);
        MPI_Scatterv(map[i+top],radi_distribute,radi_offset,MPI_DOUBLE,local_r_map[i],radi_chunk,MPI_DOUBLE,2,MPI_COMM_WORLD);
    }


    // Data use to report
    double peak_temp=30; // peak temparature for lab1, use on root
    int cont_area=0; // contaminated area for lab2, use on root
    double local_peak_temp=30; // peak temparature for lab1
    int local_cont_area=0; // contaminated area for lab2
    double sw_radius=0;
    int sw_area=0; //  shockwave blast affected area, radius for lab3
    

    // Buffer use to exchange data
    double* left_snd;
    double* right_snd;
    double* left_rcv;
    double* right_rcv;
    if(world_rank!=0){
        left_snd=(double*) malloc(sizeof(double)*y_size);
        right_snd=(double*) malloc(sizeof(double)*y_size);
        left_rcv=(double*) malloc(sizeof(double)*y_size);
        right_rcv=(double*) malloc(sizeof(double)*y_size);
        // Initialize as padding
        if(world_rank%2==1){
            for(int i=0;i<y_size;i++){
                left_snd[i]=30;
                right_snd[i]=30;
                left_rcv[i]=30;
                right_rcv[i]=30;
            }
        }
        else{
            for(int i=0;i<y_size;i++){
                left_snd[i]=0;
                right_snd[i]=0;
                left_rcv[i]=0;
                right_rcv[i]=0;
            }
        }
    }
    

    // Algorithm
    int n = 0;
    bool done = false; // Stop signal
    // Root init for lab 3
    std::vector<Worker*> workers; // Only valid on root
    Map sw_map(MAP_SIZE,INIT,map);
    if(world_rank==0){
        for(int i =0; i<NUM_THREAD;++i)
            workers.push_back(new Worker(i,&sw_map));
    }
    // Init report_req
    MPI_Ireduce(&local_peak_temp,&peak_temp,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD,&report_req[0]);
    MPI_Ireduce(&local_cont_area,&cont_area,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD,&report_req[1]);
    MPI_Ibcast(&done,1,MPI_C_BOOL,0,MPI_COMM_WORLD,&stop_sig);
    if(world_rank==0){
        MPI_Waitall(2,report_req,MPI_STATUS_IGNORE);
        std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~0~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
    }
    while(!done){
        if(world_rank==0){
            // 1 iteration of Lab 3
            shockWaveBlast(&sw_map,sw_radius,sw_area);
        }
        else if(world_rank%2==1){
            // Odd-rank processes do 1 iteration of lab 1
            // Data exchange between neighbors
            for(int i=0;i<y_size;i++){
                left_snd[i]=local_h_map[i][0];
                right_snd[i]=local_h_map[i][heat_chunk-1];
            }
            // Step 1 : from 4n+1 to 4n+3 (right_snd)
            if(world_rank%4==1 && world_rank<world_size-2){
                MPI_Isend(right_snd,y_size,MPI_DOUBLE,world_rank+2,0,MPI_COMM_WORLD,&req[0]);
            }else if(world_rank%4==3){
                MPI_Irecv(left_rcv,y_size,MPI_DOUBLE,world_rank-2,0,MPI_COMM_WORLD,&req[2]);
            }
            // Step 2 : from 4n+1 to 4n+3 (left_snd)
            if(world_rank%4==1 && world_rank>1){
                MPI_Isend(left_snd,y_size,MPI_DOUBLE,world_rank-2,0,MPI_COMM_WORLD,&req[1]);
            }else if(world_rank%4==3 && world_rank<world_size-2){
                MPI_Irecv(right_rcv,y_size,MPI_DOUBLE,world_rank+2,0,MPI_COMM_WORLD,&req[3]);
            }
            // Step 3 : from 4n+3 to 4n+1 (right_snd)
            if(world_rank%4==3 && world_rank<world_size-2){
                MPI_Isend(right_snd,y_size,MPI_DOUBLE,world_rank+2,0,MPI_COMM_WORLD,&req[0]);
            }else if(world_rank%4==1 && world_rank>1){
                MPI_Irecv(left_rcv,y_size,MPI_DOUBLE,world_rank-2,0,MPI_COMM_WORLD,&req[2]);
            }
            // Step 4 : from 4n+3 to 4n+1 (left_snd)
            if(world_rank%4==3){
                MPI_Isend(left_snd,y_size,MPI_DOUBLE,world_rank-2,0,MPI_COMM_WORLD,&req[1]);
            }else if(world_rank<world_size-2){
                MPI_Irecv(right_rcv,y_size,MPI_DOUBLE,world_rank+2,0,MPI_COMM_WORLD,&req[3]);
            }
            MPI_Waitall(4,req,MPI_STATUS_IGNORE);

            // Synchronize for the beforehand iteration Ireduce
            MPI_Waitall(2,report_req,MPI_STATUS_IGNORE);
            MPI_Wait(&stop_sig,MPI_STATUS_IGNORE);          

            // Each process use multi-thread to do 1 iteration
            heatDiffusion(local_h_map,heat_chunk,y_size,left_rcv,right_rcv,local_peak_temp);
        }
        else{
            // Even-rank processes do 1 iteration of lab 2
            // Data exchange between neighbors
            for(int i=0;i<y_size;i++){
                left_snd[i]=local_r_map[i][0];
                right_snd[i]=local_r_map[i][radi_chunk-1];
            }
            // Step 1 : from 4n+2 to 4n+4 (right_snd)
            if(world_rank%4==2 && world_rank<world_size-2){
                MPI_Isend(right_snd,y_size,MPI_DOUBLE,world_rank+2,0,MPI_COMM_WORLD,&req[0]);
            }else if(world_rank%4==0){
                MPI_Irecv(left_rcv,y_size,MPI_DOUBLE,world_rank-2,0,MPI_COMM_WORLD,&req[2]);
            }
            // Step 2 : from 4n+2 to 4n+4 (left_snd)
            if(world_rank%4==2 && world_rank>2){
                MPI_Isend(left_snd,y_size,MPI_DOUBLE,world_rank-2,0,MPI_COMM_WORLD,&req[1]);
            }else if(world_rank%4==0 && world_rank<world_size-2){
                MPI_Irecv(right_rcv,y_size,MPI_DOUBLE,world_rank+2,0,MPI_COMM_WORLD,&req[3]);
            }
            // Step 3 : from 4n+4 to 4n+2 (right_snd)
            if(world_rank%4==0 && world_rank<world_size-2){
                MPI_Isend(right_snd,y_size,MPI_DOUBLE,world_rank+2,0,MPI_COMM_WORLD,&req[0]);
            }else if(world_rank%4==2 && world_rank>2){
                MPI_Irecv(left_rcv,y_size,MPI_DOUBLE,world_rank-2,0,MPI_COMM_WORLD,&req[2]);
            }
            // Step 4 : from 4n+4 to 4n+2 (left_snd)
            if(world_rank%4==0){
                MPI_Isend(left_snd,y_size,MPI_DOUBLE,world_rank-2,0,MPI_COMM_WORLD,&req[1]);
            }else if(world_rank<world_size-2){
                MPI_Irecv(right_rcv,y_size,MPI_DOUBLE,world_rank+2,0,MPI_COMM_WORLD,&req[3]);
            }
            MPI_Waitall(4,req,MPI_STATUS_IGNORE);

            // Synchronize for the beforehand iteration Ireduce
            MPI_Waitall(2,report_req,MPI_STATUS_IGNORE);
            MPI_Wait(&stop_sig,MPI_STATUS_IGNORE);

            // Each process use multi-thread to do 1 iteration
            radiDiffusion(local_r_map,radi_chunk,y_size,left_rcv,right_rcv,local_cont_area);
            
        }
        
        
        // Reduce to get report
        MPI_Ireduce(&local_peak_temp,&peak_temp,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD,&report_req[0]);
        MPI_Ireduce(&local_cont_area,&cont_area,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD,&report_req[1]);
        if(world_rank==0){
            MPI_Waitall(2,report_req,MPI_STATUS_IGNORE);
            std::cout<<"HEAT DIFFUSION REPORT: "<<peak_temp<<std::endl;
            std::cout<<"RADI DIFFUSION REPORT: "<<cont_area<<std::endl;
            std::cout<<"SHOCKWAVE REPORT: "<<sw_radius<<" "<<sw_area<<std::endl;
            std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<n+1<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
            // Report, Check number of iterations. If 100 iterations, Broadcast stop signal
            n++;
            if(n==ITERATIONS){
                done=true;
            }
        }
        MPI_Ibcast(&done,1,MPI_C_BOOL,0,MPI_COMM_WORLD,&stop_sig);
    }

    // Post-processing 
    // Gather data
    for(int i=0;i<y_size;i++){
        MPI_Gatherv(local_h_map[i],heat_chunk,MPI_DOUBLE,map[i+top],heat_distribute,heat_offset,MPI_DOUBLE,1,MPI_COMM_WORLD);
        MPI_Gatherv(local_r_map[i],radi_chunk,MPI_DOUBLE,map[i+top],radi_distribute,radi_offset,MPI_DOUBLE,2,MPI_COMM_WORLD);
    }
    // Write data to output file
    if(world_rank==0){
        // Open a file stream for writing
        std::ofstream outputFile(LAB3);

        // Check if the file was opened successfully
        if (!outputFile.is_open()) {
            std::cerr << "Error opening "<< LAB3 << std::endl;
            return 1; // Indicate an error
        }
        for(int i = 0;i<MAP_SIZE;i++){
            for(int j = 0;j<MAP_SIZE-1;j++){
                outputFile << map[i][j+1];
                outputFile << ",";
            }
            outputFile << map[i][MAP_SIZE-1];
            outputFile << "\n";
        }
        outputFile.close();
    }else if(world_rank==1){
        // Open a file stream for writing
        std::ofstream outputFile(LAB1);

        // Check if the file was opened successfully
        if (!outputFile.is_open()) {
            std::cerr << "Error opening " << LAB1 <<std::endl;
            return 1; // Indicate an error
        }
        for(int i = 0;i<MAP_SIZE;i++){
            for(int j = 0;j<MAP_SIZE-1;j++){
                outputFile << map[i][j+1];
                outputFile << ",";
            }
            outputFile << map[i][MAP_SIZE-1];
            outputFile << "\n";
        }
        outputFile.close();
    }else if(world_rank==2){
        // Open a file stream for writing
        std::ofstream outputFile(LAB2);

        // Check if the file was opened successfully
        if (!outputFile.is_open()) {
            std::cerr << "Error opening "<< LAB2 << std::endl;
            return 1; // Indicate an error
        }
        for(int i = 0;i<MAP_SIZE;i++){
            for(int j = 0;j<MAP_SIZE-1;j++){
                outputFile << map[i][j+1];
                outputFile << ",";
            }
            outputFile << map[i][MAP_SIZE-1];
            outputFile << "\n";
        }
        outputFile.close();
    }
    
    // End timer
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if(world_rank==0){
        // Calculate duration in seconds
        double duration = end-start;

        // Print the execution time
        std::cout << "Execution time: " << duration << " seconds" << std::endl;
    }

    // Free array
    if(world_rank%2==1){
        for(int i=0;i<y_size;i++){
            free(local_h_map[i]);
            local_h_map[i]=nullptr;
        }
    }else if(world_rank!=0){
        for(int i=0;i<y_size;i++){
            free(local_r_map[i]);
            local_r_map[i]=nullptr;
        }
    }else{
        //Shut down all workers
        for (Worker* worker : workers) {
            worker->exit();
            delete worker;
        }
    }
    free(heat_distribute);
    heat_distribute=nullptr;
    free(radi_distribute);
    radi_distribute=nullptr;
    free(heat_offset);
    heat_offset=nullptr;
    free(radi_offset);
    radi_offset=nullptr;
    free(local_h_map);
    local_h_map=nullptr;
    free(local_r_map);
    local_r_map=nullptr;
    if(world_rank<=2){
        for(int i=0;i<MAP_SIZE;i++){
            free(map[i]);
            map[i]=nullptr;
        }
        free(map);
        map=nullptr;
    }
    

    MPI_Finalize();
    return 0;
}