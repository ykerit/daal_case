#include "daal.h"
#include "service.h"
#include <fstream>
#include <iostream>
#include <assert.h>

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::gbt::regression;

void load_data(NumericTablePtr& test_data) {
	float data[27 * 2] = {1.0};
	test_data = HomogenNumericTable<>::create(data, 27, 2);
}

void test_model() {
	NumericTablePtr testData;
	load_data(testData);

    prediction::Batch<float> algorithm;

	auto model_path = "./bst_wf_test.txt";
	std::ifstream file(model_path, std::ios::in|std::ios::binary|std::ios::ate);
    if(!file) {
		std::cout << "loading model error" <<std::endl;
        assert(false);
	}
	size_t length = file.tellg();
	file.seekg(0, std::ios::beg);
	byte* buffer = new byte[length];
	file.read((char*)buffer, length);
    file.close();

	OutputDataArchive out_dataArch(buffer, length);
	delete[] buffer;
	daal::algorithms::gbt::regression::ModelPtr deserialized_model = daal::algorithms::gbt::regression::Model::create(27);
	deserialized_model->deserialize(out_dataArch);
	std::cout << deserialized_model->numberOfTrees() << std::endl;
	std::cout << deserialized_model->getNumberOfFeatures() << std::endl;

    algorithm.input.set(prediction::data, testData);
    algorithm.input.set(prediction::model, deserialized_model);

    algorithm.compute();

    prediction::ResultPtr predictionResult = algorithm.getResult();
    printNumericTable(predictionResult->get(prediction::prediction), "Gragient boosted trees prediction results (first 10 rows):", 10);
}

int main()
{
	test_model();
    return 0;
}
