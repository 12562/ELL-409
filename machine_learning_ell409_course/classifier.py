import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab   as mlab
import csv
import sys

def data_specific(data, labels, csv_reader):
    line_count = 0
    label_col  = 0
    for row in csv_reader:
        row_to_append = []
        if line_count == 0:
           for col in row:
               if 'output=' in col:
                  label_col = row.index(col)
               #n = len(row)
               #if "output=" in row
        else:
           for i in range(len(row)):    
               if ( i != label_col ):
                  row_to_append.append(row[i])
              #if row[4] == 'FIRST_AC':
              #   seating_pref = 1
              #elif row[4] == 'SECOND_AC':
              #   seating_pref = 2
              #elif row[4] == 'THIRD_AC':
              #   seating_pref = 3
              #else:
              #   seating_pref = 0
              # 
              #if row[5] == 'male':
              #   person = 0
              #else:
              #   person = 1
              # 
              #data.append([row[0],row[2],row[3],seating_pref,person,row[6]])
           data.append(row_to_append)
           labels.append(row[label_col])
        line_count += 1


def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b

def map_string_to_num(data_array, array):
    column_unique_ele = []
    if ( array ):
       for index in range(len(data_array[0])):
           if ( data_array[0,index][0].isalpha() ):
              column_unique_ele = np.array(np.unique(data_array[:,index]))
              num = len(column_unique_ele)
              for i in range(0,np.shape(data_array)[0]):
                  data_array[i,index] = np.where(column_unique_ele == data_array[i,index])[0][0]
       return(data_array.tolist())
    else:
       if ( data_array[0].isalpha() ):
          column_unique_ele = np.array(np.unique(data_array[:]))
          num = len(column_unique_ele)
          for i in range(0,np.shape(data_array)[0]):
              data_array[i] = np.where(column_unique_ele == data_array[i])[0][0]
       return(data_array.tolist())
               

def get_data(file_name,inp_data_type, out_data_type):
    global inpdata,data,labels,train_m,train_n,test_m,test_n,train_data,test_data,train_labels,test_labels
    data = []
    labels = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data_specific(data, labels, csv_reader)
    inpdata = np.array(map_string_to_num(np.array(data), 1)).astype(inp_data_type)
    labels  = np.array(map_string_to_num(np.array(labels),  0)).astype(out_data_type)
    total_m = np.shape(inpdata)[0]
    test_m  = total_m / 6
    train_m = total_m - test_m
    train_n = np.shape(inpdata)[1]
    test_n  = train_n
    train_data = inpdata[0:train_m]
    test_data  = inpdata[train_m:train_m+test_m]
    train_labels = labels[0:train_m]
    test_labels = labels[train_m:train_m+test_m]


get_data(sys.argv[2], sys.argv[3], sys.argv[4])

def get_mean_mode_std_data():
    global labels, num_classes, class_means, class_modes, class_std, means_list, modes_list, std_list, train_data, train_labels 
    label_names = np.unique(labels)
    num_classes = np.shape(label_names)[0]
    class_means   = list()
    class_modes   = list()
    class_std    = list()
    for i in range(0,num_classes):
       class_means.append(np.mean(train_data[train_labels==label_names[i]],0).tolist())
       class_modes.append(max(train_data[train_labels==label_names[i]].tolist(), key=train_data[train_labels==label_names[i]].tolist().count))
       class_std.append(np.std(train_data[train_labels==label_names[i]],0).tolist())
    #parzenwindow_means_list = np.array(np.reshape(np.zeros(num_classes * train_n),[num_classes, train_n]))
    #parzenwindow_std_list   = np.array(np.reshape(np.ones (num_classes * train_n),[num_classes, train_n]))
    means_list              = np.array(class_means)
    modes_list              = np.array(class_modes)
    std_list                = np.array(class_std)

regtype = sys.argv[5]
if ( regtype != "regression" ):
   get_mean_mode_std_data()

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
### K means clustering with initial mu values according to mean of train labels available

def k_means_clustering(num_classes, train_data, train_labels):
    global means_list
    k_means_classified_labels = []
    # Initial mu matrix according to training labels to compute accuracy
    cluster_mu_matrix = means_list
    clusters = []
    samples, num_features = np.shape(train_data)
    total_mu_values = num_features * num_classes
    #cluster_mu_matrix = np.reshape(np.zeros(total_mu_values),[num_classes,num_features])
    for i in range(0,num_classes):
        clusters.append([])
    for j in range(0,samples):
        x        = train_data[j]
        rep_x    = np.repeat(np.reshape(x,[1,num_features]), num_classes, 0)
        distance = np.reshape(np.sum(np.power(rep_x - cluster_mu_matrix, 2), 1), [num_classes,1])
        class_assigned = min(np.argmin(distance, axis=0))
        clusters[class_assigned].append(x)
        cluster_mu_matrix[class_assigned] = np.mean(clusters[class_assigned], axis=0)
        k_means_classified_labels.append(class_assigned)
    return k_means_classified_labels
#################################################################################
#k_means_clustering_init(num_classes, train_data, train_labels)
#
#def k_means_clustering(data, labels, cluster_mu_matrix):
#    global cluster_mu_matrix
#    samples, num_features = np.shape(data)
#    k_means_classified_labels = []
#    for j in range(0,samples):
#        x        = data[j]
#        rep_x    = np.repeat(np.reshape(x,[1,num_features]), num_classes, 0)
#        distance = np.reshape(np.sum(np.power(rep_x - cluster_mu_matrix, 2), 1), [num_classes,1])
#    
#        k_means_classified_labels.append(class_assigned)
################################################################################
#def parzen_window_classifier(num_classes, train_labels, train_data, train_data_m, train_data_n, test_data, test_labels, h ):
#        parzenwindow_std_list = np.array(np.reshape(np.ones(train_m)),[1,train_m])
#        parzenwindow_means_list = np.array(np.reshape(np.zeros(train_m)),[1,train_m])
#        x_transpose = np.chararray.transpose(train_data)
#        sigma    = parzenwindow_std_list
#        mu       = parzenwindow_means_list
#        for i in range(0,train_n):
#            X        = x_transpose[i]
#            cnst     = 1 / np.sqrt(np.sqrt(2*np.pi) * sigma)
#            power    = - pow((X - mu), 2) / ( 2 * pow(sigma,2))
#            gaussian = cnst * pow(np.e, power)
#            np.copyto(gaussian,mu,casting='same_kind', where=np.bitwise_not(np.isfinite(gaussian)))
#            res_gauss.append(gaussian.tolist())
#        res_gauss = np.chararray.transpose(res_gauss)
#        density_array =   
################################################################################
### Parzen window based classifier with window function as a gaussian
def parzen_window_classifier(num_classes, train_labels, train_data, test_data, test_labels, h ):
    classified_labels = []
    train_data_m, train_data_n = np.shape(train_data)
    test_data_m , test_data_n  = np.shape(test_data)
    d = train_data_n
    for j in range(0,test_data_m):
        res_gauss = []
        start_X = test_data[j]
        for i in range(0,num_classes):
            #class_m    = np.size(train_labels) - np.size(np.nonzero(train_labels - i))
            X_i        = train_data[train_labels == i]
            class_m    = np.shape(X_i)[0]
            X          = np.repeat(np.array([start_X]), class_m, 0)
            prior = (100.0 * class_m) / train_data_m
            cnst       = 1 / (pow( h * np.sqrt(2 * np.pi), d) * class_m)
            power      = - ( 1.0 * ((X - X_i) * (X - X_i))) / (2 * h * h)
            gaussian   = cnst * pow(np.e, power)
            likelihood      = np.sum(gaussian, 0)
            posterior  = likelihood * prior
            res_gauss.append(posterior.tolist())
        classified_labels.append(max(np.argmax(res_gauss, axis=0), key=np.argmax(res_gauss, axis=0).tolist().count))
    return classified_labels
################################################################################
### knn classifer (k = closest neighbours)
def kargmin(inparr, k):
    k_argmin_index = []
    for i in range(0,k):
        k_argmin_index.append(np.where(np.argsort(inparr) == i)[0][0])
    return np.array(k_argmin_index)

def knn_classifier(train_data, train_labels, test_data, test_labels, k):
    knn_classified_labels = []
    (train_m, train_n) = np.shape(train_data);
    (test_m , test_n ) = np.shape(test_data);
    for i in range(0,test_m):
        repeated_test_point_matrix = np.repeat(np.reshape(test_data[i],[1,test_n]), train_m ,0)
        distance_vector = np.sqrt(np.sum(np.power(repeated_test_point_matrix - train_data, 2), 1))
        knn = kargmin(distance_vector, k)
        knn_classified_labels.append(max(train_labels[knn].tolist(), key=train_labels[knn].tolist().count))
    return knn_classified_labels
################################################################################
### Linear model using gradient descent(k = No. of iterations, n = Learning rate)
def learn_weights(weights, X, labels, learning_rate, num_iterations):
    m,n = np.shape(X)
    for i in range(0,num_iterations):
        h         = np.dot(X, np.transpose(weights))
        diff      = np.transpose(h - np.reshape(labels, [m,1]))
        summation = np.dot(diff, X)
        weights   = weights - ((learning_rate * summation) / m)
    return weights
        #weights = weights - (learning_rate * (np.dot(np.transpose(np.reshape(np.dot(X, np.transpose(weights)),[m,1]) - np.reshape(labels, [m,1])),  X) / m))
     
#def sigmoid(h):
#    z   = -h
#    return ( 1 / ( 1 + pow(np.e, z) )

def linear_regression(n, k, train_data, train_labels, test_data, test_labels):
    (train_m, train_n) = np.shape(train_data)
    (test_m , test_n ) = np.shape(test_data)
    X    = train_data
    X    = np.insert(X, 0, 1, axis=1)
    W    = np.zeros((1, test_n + 1))
    W    = learn_weights(W, X, train_labels, n, k) 
    X    = np.insert(test_data, 0, 1, axis=1)
    predicted_labels = np.transpose(np.dot(X, np.transpose(W)))
    return predicted_labels
    
################################################################################
def learn_logistic_weights(weights, X, labels, learning_rate, num_iterations):
    m,n = np.shape(X)
    for i in range(0,num_iterations):
        h         = np.dot(X, np.transpose(weights))
        diff      = np.transpose(h - np.reshape(labels, [m,1]))
        summation = np.dot(diff, X)
        weights   = weights - ((learning_rate * summation) / m)
    return weights

def logistic_regression(alpha, n):
    (train_m, train_n) = np.shape(train_data)
    (test_m , test_n ) = np.shape(test_data)
    X = train_data
    X = np.insert(X, 0, 1, axis=1)
    
    
      
################################################################################

samples = train_m
num_features = train_n

if ( regtype != "regression" ):
   ######################################################################################################################
   ### Baye's classifier on training data with mle
   classified_labels = bayes_classifier(samples, num_features, num_classes, train_data, train_labels, "mle" ) 
   true_labels = np.unique(train_labels, return_inverse=True)[1]
    
   accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
   print "\n\nAccuracy for Baye's classifier with parameters estimated using mle on training data: %d %% " % accuracy
    
   samples = test_m
   ### Baye's classifier on test data with mle
   classified_labels = bayes_classifier(samples, num_features, num_classes, test_data, test_labels, "mle" ) 
   true_labels = np.unique(test_labels, return_inverse=True)[1]
    
   accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
   print "Accuracy for Baye's classifier with parameters estimated using mle on test data: %d %% \n" % accuracy
   ######################################################################################################################
    
    
   ######################################################################################################################
   samples = train_m
   ### Baye's classifier on training data with map
   classified_labels = bayes_classifier(samples, num_features, num_classes, train_data, train_labels, "map" ) 
   true_labels = np.unique(train_labels, return_inverse=True)[1]
   accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
   print "\n\nAccuracy for Baye's classifier with parameters estimated using map on train data: %d %% " % accuracy
    
    
   samples = test_m
   ### Baye's classifier on test data with map
   classified_labels = bayes_classifier(samples, num_features, num_classes, test_data, test_labels, "map" ) 
   true_labels = np.unique(test_labels, return_inverse=True)[1]
   accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
   print "Accuracy for Baye's classifier with parameters estimated using map on test data: %d %% \n" % accuracy
   ######################################################################################################################
    
    
   ######################################################################################################################
   samples = train_m
   ### Naive Baye's classifier on training data with map
   classified_labels = naive_bayes_classifier(samples, num_features, num_classes, train_data, train_labels, "mle")
   true_labels = np.unique(train_labels, return_inverse=True)[1]
   accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
   print "\n\nAccuracy for Naive Baye's classifier with parameters estimated using mle on train data: %d %% " % accuracy
    
   samples = test_m
   ### Naive Baye's classifier on test data with map
   classified_labels = naive_bayes_classifier(samples, num_features, num_classes, test_data, test_labels, "mle")
   true_labels = np.unique(test_labels, return_inverse=True)[1]
   accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
   print "Accuracy for Naive Baye's classifier with parameters estimated using mle on test data: %d %% \n" % accuracy
   ######################################################################################################################
    
   samples = train_m
   ### K-means clustering classifier(with cluster center as train data mean) on train data
   classified_labels = k_means_clustering(num_classes, train_data, train_labels)
   true_labels = np.unique(train_labels, return_inverse=True)[1]
   accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
   print "\n\nAccuracy for nearest neighbour estimate based classifier on train data (for initial cluster center taken as mean of train data) : %d %% " % accuracy
    
   samples = test_m
   ### K-means clustering classifier(with cluster center as train data mean) on test data
   classified_labels = k_means_clustering(num_classes, test_data, test_labels)
   true_labels = np.unique(test_labels, return_inverse=True)[1]
   accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
   print "Accuracy for nearest neighbour estimate based classifier on train data (for initial cluster center taken as mean of train data) : %d %% \n" % accuracy
   ######################################################################################################################
    
   #samples = test_m
   #### Parzen window estimate based classifier on test  data
   #classified_labels = parzen_window_classifier(num_classes, train_labels, train_data, test_data, test_labels, h)
   #true_labels = np.unique(test_labels, return_inverse=True)[1]
   #accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
   #print "Accuracy for Parzen window estimate based classifier on test data (for h = %f) : %d %% \n\n\n" % (h, accuracy)
   ######################################################################################################################
   ######################################################################################################################
   hypercube_width = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
   samples = train_m
   ### Parzen window estimate based classifier on train data
   for h in hypercube_width:
       classified_labels = parzen_window_classifier(num_classes, train_labels, train_data, train_data, train_labels, h)
       true_labels = np.unique(train_labels, return_inverse=True)[1]
       accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
       print "Accuracy for Parzen window estimate based classifier on train data (for h = %f) : %d %% " % (h, accuracy)
    
   print "\n"
   samples = test_m
   ### Parzen window estimate based classifier on test  data
   for h in hypercube_width:
       classified_labels = parzen_window_classifier(num_classes, train_labels, train_data, test_data, test_labels, h)
       true_labels = np.unique(test_labels, return_inverse=True)[1]
       accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
       print "Accuracy for Parzen window estimate based classifier on test data (for h = %f) : %d %% " % (h, accuracy)
   ######################################################################################################################
   nearest_neighbours = [1000, 500, 100, 75, 50, 25, 10, 7, 5, 3, 2, 1]
   samples = train_m
   ###  based classifier on train data
   for k in nearest_neighbours:
       classified_labels = knn_classifier(train_data, train_labels, train_data, train_labels, k)
       true_labels = np.unique(train_labels, return_inverse=True)[1]
       accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
       print "Accuracy for knn estimate based classifier on train data (for k = %d) : %d %% " % (k, accuracy)
    
   print "\n"
   samples = test_m
   ### Parzen window estimate based classifier on test  data
   for k in nearest_neighbours:
       classified_labels = knn_classifier(train_data, train_labels, test_data, test_labels, k)
       true_labels = np.unique(test_labels, return_inverse=True)[1]
       accuracy = ((samples - np.size(np.nonzero(np.array(classified_labels) - true_labels[0:samples]))) * 100) / samples
       print "Accuracy for knn estimate based classifier on test data (for k = %d) : %d %% " % (k, accuracy)
   ######################################################################################################################

def accuracy_regression(train_data, train_labels, data, labels):
    iter = [ 10, 100, 1000, 10000, 100000]
    rate = [ 0.1, 0.01, 0.001, 0.0001, 0.00001]
    for k in iter:
        for n in rate:
            predicted_labels = linear_regression(n, k, train_data, train_labels, data, labels, "linear")
            average_change =  np.mean(100 * ((predicted_labels - labels) / labels))
            print "Average change for iterations = %d and learning rate = %f : %f" % (k, n, average_change)

if regtype == "regression":
   accuracy_regression(train_data, train_labels, train_data, train_labels)
   accuracy_regression(train_data, train_labels, test_data,  test_labels)

