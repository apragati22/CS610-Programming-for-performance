#include<bits/stdc++.h>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
using namespace std;

ifstream inputFile; 
ofstream outputFile;
atomic<int> totalBytesRead = 0; // for seeking in input file for each producer
mutex producerLock; // exclusive producer lock for inorder writing to buffer
mutex prodConsLock; // lock shared by producers and consumers for buffer access
mutex inputFileLock; // lock for reading from input file
mutex printLock; // lock for printing to console (for debugging only)
vector<string> buffer; // shared buffer for producers and consumers (size M)
condition_variable bufferStatus; // Condition variable to signal when buffer has data or data is read
atomic<bool> taskCompleted(false); // flag to indicate if all producers have finished writing to the buffer
atomic<int> producersDone(0); // count of producers that have finished writing

void producerThreadFunction(const int& producerId, const int& Lmax, const int& Lmin, const int& bufferSize, const int& numProducerThreads) {
    string line;
    vector<string> linesToWrite;
    while (true) {
        // Randomly determine how many lines to read
        int L = rand() % (Lmax - Lmin + 1) + Lmin;

        unique_lock<mutex> fileLock(inputFileLock);

        // Seek to the last read position for this producer
        inputFile.seekg(totalBytesRead, ios::beg); 

        for (int i = 0; i < L && getline(inputFile, line); ++i) {
            totalBytesRead += line.size() + 1; // +1 for newline character
            linesToWrite.push_back(line);
        }
        
        // unique_lock<mutex> printLockGuard(printLock);
        // cout<<"Producer " << producerId << " read " << linesToWrite.size() << " lines." << endl;
        // printLockGuard.unlock();
        
        fileLock.unlock();

        if (linesToWrite.empty()) { // file has no more lines to read
            if (++producersDone == numProducerThreads) {
               taskCompleted.store(true); 
                unique_lock<mutex> lock(prodConsLock);
                bufferStatus.notify_all(); // Notify consumers that no more lines will be produced
            }
            break; 
        }

        // Write to the buffer
        {
            unique_lock<mutex> prodLock(producerLock);
            int writtenLines = 0;
            while(writtenLines < linesToWrite.size()) {
                unique_lock<mutex> bufferLock(prodConsLock);

                // unique_lock<mutex> printLockGuard(printLock);
                // if(buffer.size()<bufferSize) cout << "Producer " << producerId << " writing to buffer." << endl;
                // else cout << "Producer " << producerId << " buffer is full, waiting." << endl;
                // printLockGuard.unlock();

                // Wait until there is space in the buffer
                bufferStatus.wait(bufferLock, [&]() { return buffer.size() < bufferSize; });
                while (writtenLines < linesToWrite.size() && buffer.size() < bufferSize) {

                    // unique_lock<mutex> printLockGuard(printLock);
                    // cout << "Producer " << producerId << " writing to buffer: " << linesToWrite[writtenLines] << endl;
                    // printLockGuard.unlock();
                    
                    buffer.push_back(linesToWrite[writtenLines]);
                    writtenLines++;
                    bufferStatus.notify_all();
                }
            }
            linesToWrite.clear();
            prodLock.unlock();
        }
    }    
}


void consumerThreadFunction(int consumerId, int bufferSize) {

    while (true) {
        vector<string> linesToWrite;
        string line;
        { 
            // Lock to access the shared buffer
            unique_lock<mutex> bufferLock(prodConsLock);
            
            if (buffer.empty() && taskCompleted.load()) {   
                break; // No more lines to consume
            }
            
            if(!buffer.empty()) {
                for(const auto& line:buffer){
                    bufferStatus.wait(bufferLock, []() { return !buffer.empty() || taskCompleted.load(); });
                    linesToWrite.push_back(line);
                    bufferStatus.notify_all();  
                }
                buffer.clear();

                // unique_lock<mutex> printLockGuard(printLock);
                // cout << "Consumer " << consumerId << " writing " <<linesToWrite.size() <<" lines." << endl;
                // printLockGuard.unlock();

                for (const auto &line : linesToWrite) {
                    outputFile << line << endl; 
                }
            }
            bufferLock.unlock();
        }
    }
    
}

int main(int argc, char *argv[]) {
    if(argc<7){
        cerr << "Usage: " << argv[0] << " <input_file> <num_producer_threads> <Lmin> <Lmax> <buffer_size> <output_file>" << endl;
        return 1;
    }
    string inputFileName = argv[1];
    int numProducerThreads = stoi(argv[2]);
    int Lmin = stoi(argv[3]);
    int Lmax = stoi(argv[4]);
    int bufferSize = stoi(argv[5]);
    string outputFileName = argv[6];

    inputFile.open(inputFileName);
    if (!inputFile.is_open()) {
        cerr << "Error opening input file: " << inputFileName << endl;
        return 1;
    }

    outputFile.open(outputFileName);
    if (!outputFile.is_open()) {
        cerr << "Error opening output file: " << outputFileName << endl;
        return 1;
    }

    int numConsumerThreads = max(1, numProducerThreads/2);
    vector<thread> producerThreads(numProducerThreads);
    vector<thread> consumerThreads(numConsumerThreads);
    
    buffer.reserve(bufferSize); 

    for(int i=0; i<numProducerThreads; i++) {
        producerThreads[i] = thread(producerThreadFunction, i, Lmax, Lmin, bufferSize, numProducerThreads);
        if (!producerThreads[i].joinable()) {
            cerr << "Error creating producer thread " << i << endl;
            return 1;
        }
    }

    for(int i=0; i<numConsumerThreads; i++) {
        consumerThreads[i] = thread(consumerThreadFunction, i, bufferSize);
        if (!consumerThreads[i].joinable()) {
            cerr << "Error creating consumer thread " << i << endl;
            return 1;
        }
    }

    for(auto &t : producerThreads) {
        t.join();
    }
    for(auto &t : consumerThreads) {
        t.join();
    }

    inputFile.close();
    outputFile.close();

    cout << "Processing complete. Output written to " << outputFileName << endl;

    // Clean up and exit
    bufferStatus.notify_all();
    buffer.clear();
    producerThreads.clear();
    consumerThreads.clear();

    return 0;
}

