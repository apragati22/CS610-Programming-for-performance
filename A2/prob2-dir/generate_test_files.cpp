#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <filesystem>

using namespace std;

// Generate a random word of length 3-12 characters
string generateRandomWord(mt19937& gen) {
    uniform_int_distribution<int> lengthDist(3, 12);
    uniform_int_distribution<int> charDist('a', 'z');
    
    int length = lengthDist(gen);
    string word;
    word.reserve(length);
    
    for (int i = 0; i < length; i++) {
        word += static_cast<char>(charDist(gen));
    }
    
    return word;
}

// Generate a file with random words until it reaches approximately targetSize bytes
void generateFile(const string& filename, size_t targetSize) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not create file " << filename << endl;
        return;
    }
    
    mt19937 gen(random_device{}());
    uniform_int_distribution<int> wordsPerLineDist(5, 15);
    
    size_t currentSize = 0;
    
    while (currentSize < targetSize) {
        int wordsInLine = wordsPerLineDist(gen);
        
        for (int i = 0; i < wordsInLine && currentSize < targetSize; i++) {
            string word = generateRandomWord(gen);
            file << word;
            currentSize += word.length();
            
            if (i < wordsInLine - 1 && currentSize < targetSize) {
                file << " ";
                currentSize += 1;
            }
        }
        
        if (currentSize < targetSize) {
            file << "\n";
            currentSize += 1;
        }
    }
    
    file.close();
    
    size_t actualSize = filesystem::file_size(filename);
    cout << "Generated " << filename << " - Size: " << actualSize << " bytes (~" 
         << (actualSize / 1024.0 / 1024.0) << " MB)" << endl;
}

int main() {
    const size_t targetSize = 4 * 1024 * 1024; 
    const string outputDir = "prob2-test1-files";
    const string inputFile = outputDir + "/input";
    
    filesystem::create_directories(outputDir);
    
    cout << "Generating 5 test files with ~4MB each..." << endl;
    
    for (int i = 1; i <= 5; i++) {
        string filename = outputDir + "/file" + to_string(i) + ".txt";
        generateFile(filename, targetSize);
    }
    
    ofstream inputFileStream(inputFile);
    if (inputFileStream.is_open()) {
        for (int i = 1; i <= 5; i++) {
            inputFileStream << "file" << i << ".txt" << endl;
        }
        inputFileStream.close();
        cout << "Created input file: " << inputFile << endl;
    } else {
        cerr << "Error: Could not create input file " << inputFile << endl;
    }
    
    cout << "All files generated successfully!" << endl;
    
    return 0;
}