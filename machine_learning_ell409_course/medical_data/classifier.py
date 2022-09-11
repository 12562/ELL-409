import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab   as mlab
import csv

def get_data():
    data = []
    labels = []
    with open('Medical_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                data.append([row[1],row[2],row[3]])
                labels.append(row[0])
            line_count += 1
    
    inpdata = np.array(data).astype(np.float)
    labels  = np.array(labels)
    train_data = inpdata[0:2000]
    test_data  = inpdata[2000:3000]
    train_labels = labels[0:2000]
    test_labels = labels[2000:3000]


label_names = np.unique(labels)
class_means   = list()
class_modes   = list()
class_std    = list()
for i in range(0,3):
   class_means.append(np.mean(train_data[train_labels==label_names[i]],0).tolist())
   class_modes.append(max(train_data[train_labels==label_names[i]].tolist(), key=train_data[train_labels==label_names[i]].tolist().count))
   class_std.append(np.std(train_data[train_labels==label_names[i]],0).tolist())

means_list        = np.array(class_means)
modes_list        = np.array(class_modes)
std_list          = np.array(class_std)

################################BAYE'S CLASSIFIER###############################
def bayes_classifier(samples, num_features, num_classes, train_data, train_labels, estimate_type):
    mle_classified_labels = []
    if ( estimate_type == "mle" ):
       global std_list, means_list
    else:
       global std_list, modes_list
    for j in range(0, samples):
        x = train_data[j]
        res_gauss = []
        for i in range(0,num_classes):
           sigma = std_list[i]
           if ( estimate_type == "mle" ): 
              mu    = means_list[i]
           else:
              mu    = modes_list[i]
           X_i   = x
           cnst     = 1 / np.sqrt(np.sqrt(2*np.pi) * sigma)
           power    = - pow((X_i - mu),2) / ( 2 * pow(sigma,2))
           gaussian = cnst * pow(np.e, power)
           np.copyto(gaussian,mu,casting='same_kind', where=np.bitwise_not(np.isfinite(gaussian)))
           res_gauss.append(gaussian.tolist())
        mle_classified_labels.append(max(np.argmax(res_gauss, axis=0), key=np.argmax(res_gauss, axis=0).tolist().count))
    return mle_classified_labels
################################################################################

##########################NAIVE BAYE's CLASSIFIER###############################
def naive_bayes_classifier(samples, num_features, num_classes, train_data, train_labels, estimation_type):
    global std_list,means_list
    naive_bayes_classified_labels = []
    for j in range(0, samples):
        x = train_data[j]
        res_gauss = []
        for i in range(0,num_classes):
            sigma = std_list[i]
            mu    = means_list[i]
            X_i   = x
            cnst     = 1 / np.sqrt(np.sqrt(2*np.pi) * sigma)
            power    = - pow((X_i - mu),2) / ( 2 * pow(sigma,2))
            gaussian = cnst * pow(np.e, power)
            np.copyto(gaussian,mu,casting='same_kind', where=np.bitwise_not(np.isfinite(gaussian)))
            gaussian = gaussian / cnst
            res_gauss.append(gaussian.tolist())
        joint_model = np.reshape(np.prod(np.array(res_gauss),axis=1),[num_classes,1]).tolist()
        naive_bayes_classified_labels.append(max(np.argmax(joint_model, axis=0), key=np.argmax(joint_model, axis=0).tolist().count))
    return naive_bayes_classified_labels

################################################################################

samples = 2000
num_features = 3
num_classes = 3

### Baye's classifier on training data with mle
classified_labels = bayes_classifier(samples, num_features, num_classes, train_data, train_labels, "mle" ) 
true_labels = np.unique(train_labels, return_inverse=True)[1]

accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
print "\n\nAccuracy for Baye's classifier with parameters estimated using mle on training data: %d %% \n" % accuracy

samples=1000
### Baye's classifier on test data with mle
classified_labels = bayes_classifier(samples, num_features, num_classes, test_data, test_labels, "mle" ) 
true_labels = np.unique(test_labels, return_inverse=True)[1]

accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
print "\n\nAccuracy for Baye's classifier with parameters estimated using mle on test data: %d %% \n" % accuracy


samples=2000
### Baye's classifier on training data with map
classified_labels = bayes_classifier(samples, num_features, num_classes, train_data, train_labels, "map" ) 
true_labels = np.unique(train_labels, return_inverse=True)[1]

accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
print "\n\nAccuracy for Baye's classifier with parameters estimated using map on train data: %d %% \n" % accuracy


samples=1000
### Baye's classifier on test data with map
classified_labels = bayes_classifier(samples, num_features, num_classes, test_data, test_labels, "map" ) 
true_labels = np.unique(test_labels, return_inverse=True)[1]

accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
print "\n\nAccuracy for Baye's classifier with parameters estimated using map on test data: %d %% \n" % accuracy


### Naive Baye's classifier on training data with map
classified_labels = naive_bayes_classifier(samples, num_features, num_classes, train_data, train_labels, "mle")

true_labels = np.unique(train_labels, return_inverse=True)[1]

accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
print "\n\nAccuracy for Naive Baye's classifier with parameters estimated using mle on train data: %d %% \n" % accuracy


### Naive Baye's classifier on test data with map
classified_labels = naive_bayes_classifier(samples, num_features, num_classes, test_data, test_labels, "mle")

true_labels = np.unique(test_labels, return_inverse=True)[1]

accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
print "\n\nAccuracy for Naive Baye's classifier with parameters estimated using mle on test data: %d %% \n" % accuracy


