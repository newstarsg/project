from __future__ import division
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import time
import datetime
import random
random.seed(9001)

from sklearn import svm
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
import sklearn.linear_model as linear_model
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
import xgboost as xgb
import pdb


def pr_curve(y_test,y_score):
	precision, recall, _ = precision_recall_curve(y_test, y_score)
	average_precision = average_precision_score(y_test, y_score)
	f1_s = f1_score(y_test, y_score>0.5, average='micro')
	fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, pos_label=1)
	auc = metrics.auc(fpr, tpr)
	return average_precision,f1_s,auc

def save_file(save_dir,save_name,x):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir + save_name,x)

def read_train(train_csv_file, userid_input):
	#get the training data, as well as the statistics
	#train_csv_file = './reactivation_data/training.csv'
	train_record = pd.read_table(train_csv_file,sep = ',')
	voucher_code_received = train_record.voucher_code_received.unique()
	promotionid_received = train_record.promotionid_received.unique()
	userid = train_record.userid.unique()
	train_record['used?'].value_counts()
	train_record['repurchase_15?'].value_counts()
	train_record['repurchase_30?'].value_counts()
	train_record['repurchase_60?'].value_counts()
	train_record['repurchase_90?'].value_counts()
	train_record['UTC Time'] = pd.to_datetime(train_record['voucher_received_time'],unit='s')
	labels = train_record[['used?','repurchase_15?','repurchase_30?','repurchase_60?','repurchase_90?']]
	train_record_cur = train_record.loc[train_record['userid'] == userid_input]
	return train_record,labels

def process_train(train_record_old,train_type,reactive_data_new,user_profiles):
	#construct the training data
	reactive_data_new = reactive_data_new.fillna(0)
	train_record = train_record_old.merge(reactive_data_new, left_on=['userid','voucher_received_time','promotionid_received','voucher_code_received'], right_on=['userid','voucher_received_time','promotionid_received','voucher_code_received'], how='inner')
	train_record['ind'] = pd.Series(range(0,train_record.shape[0]), index=train_record.index)
	train_record = train_record.merge(user_profiles,left_on='userid',right_on = 'userid',how = 'inner')
	train_record = train_record.sort_values(by=['ind'])
	train_record = train_record.drop('ind',1)
	if train_type == 1:
		column_select = ['userid','promotionid_received','voucher_code_received','voucher_received_time','used?','repurchase_15?','repurchase_30?','repurchase_60?','repurchase_90?','UTC Time']
	else:
		column_select = ['userid','promotionid_received','voucher_code_received','voucher_received_time']
	train_record_tem = train_record.drop(column_select, axis=1)
	train_record_new = train_record_tem

	column_names = train_record_new.columns
	return train_record_new


def read_test(train_csv_file):
	#get the training data, as well as the statistics
	#train_csv_file = './reactivation_data/training.csv'
	train_record = pd.read_table(train_csv_file,sep = ',')
	voucher_code_received = train_record.voucher_code_received.unique()
	promotionid_received = train_record.promotionid_received.unique()
	userid = train_record.userid.unique()
	return train_record

def read_transaction(userid_input):
	# lookup the transaction history for each user
	trnasactions_csv_file = './reactivation_data/transactions_MY.csv'
	transactions_MY = pd.read_table(trnasactions_csv_file,sep = ',')
	userid = transactions_MY.userid.unique()
	shopid = transactions_MY.shopid.unique()
	userid_cur = userid_input
	tran_cur = transactions_MY.loc[transactions_MY['userid'] == userid_cur]
	pdb.set_trace()
	return tran_cur # get the transaction for the specific user

def read_user(userid_input):
	# get the user profile
	user_profile_csv_file = './reactivation_data/user_profiles_MY.csv'
	user_profiles = pd.read_table(user_profile_csv_file,sep = ',')
	userid = user_profiles.userid.unique()
	gender = user_profiles.gender.unique()
	userid_cur = userid_input#12956#userid[0]
	user_cur = user_profiles.loc[user_profiles['userid'] == userid_cur]
	column_select = ['registration_time','birthday']
	user_profiles = user_profiles.drop(column_select, axis=1)
	user_profiles = user_profiles.fillna({'gender':0})
	user_gender_enc = user_profiles['gender'].apply(str)
	user_gender_enc = pd.get_dummies(user_gender_enc,prefix = ["usr_gen"])
	user_profiles = pd.concat([user_profiles, user_gender_enc], axis=1)
	column_select = ['gender']
	user_profiles = user_profiles.drop(column_select, axis=1)
	return user_profiles

def read_likes(userid_input):
	likes_csv_file = './reactivation_data/likes.csv'
	user_likes = pd.read_table(likes_csv_file,sep = ',')
	userid = user_likes.userid.unique()
	itemid = user_likes.itemid.unique()
	userid_cur = userid_input#userid[0]
	user_cur = user_likes.loc[user_likes['userid'] == userid_cur]
	return user_cur
	pdb.set_trace()
def read_view_history(userid_input):
	pre_day = 1
	print pre_day
	view_log_csv_file = './reactivation_data/view_log_'+str(pre_day)+'.csv'
	view_log = pd.read_table(view_log_csv_file,sep = ',')
	userid = view_log.userid.unique()
	event_name = view_log.event_name.unique()
	userid_cur = userid_input#12956#userid[0]
	user_cur = view_log.loc[view_log['userid'] == userid_cur]
	return user_cur

def read_reactive_data(userid_input):
	reactive_csv_file = './reactivation_data/voucher_distribution_active_date.csv'
	reactive_data = pd.read_table(reactive_csv_file,sep = ',')
	userid = reactive_data.userid.unique()
	promotionid_received = reactive_data.promotionid_received.unique()
	voucher_code_received = reactive_data.voucher_code_received.unique()

	obj_train_record = reactive_data['voucher_code_received']
	train_record_vouid = pd.get_dummies(obj_train_record, prefix = ["vou_id"])
	train_record_proid = reactive_data['promotionid_received'].apply(str)
	train_record_proid = pd.get_dummies(train_record_proid,prefix = ["pro_id"])
	train_record_new = pd.concat([reactive_data,train_record_vouid, train_record_proid], axis=1)

	column_select = ['voucher_received_date']
	reactive_data_new = train_record_new.drop(column_select, axis=1)
	return reactive_data_new

def read_voucher_mechanics():
	voucher_mechanics_csv_file = './reactivation_data/voucher_mechanics.csv'
	voucher_mechanics = pd.read_table(voucher_mechanics_csv_file,sep = ',')
	return voucher_mechanics
user_id = 13062#995583#13014#13098
train_csv_file = './reactivation_data/training.csv'
# train_info = read_train(train_csv_file,user_id)
# print train_info
# test_csv_file = './reactivation_data/predict.csv'
# test_info = read_test(test_csv_file,user_id)
# print test_info
# tran_info = read_transaction(user_id)
# print tran_info
user_profiles = read_user(user_id)
# print user_cur
# user_likes = read_likes(user_id)
# print user_likes
# user_view_history = read_view_history(user_id)
# print user_view_history
# reactive_data_new = read_reactive_data(user_id)
# print user_reactive_hist
# voucher_meachanics = read_voucher_mechanics()

train_record,labels = read_train(train_csv_file,user_id)
reactive_data_new = read_reactive_data(user_id)

train_record_new = process_train(train_record,1,reactive_data_new,user_profiles)

validation = 0

if validation == 1:
	train, test,labels_tr,labels_te = train_test_split(train_record_new,labels, test_size=0.2,random_state=42)

	algo = 1
	X = train
	y = labels_tr['used?']
	y_te = labels_te['used?']

	if algo == 1: # LR
		lr = linear_model.LogisticRegression(C=10)
		lr.fit(X, y)
		y_pred_te = lr.predict_proba(test)
		y_pred_tr = lr.predict_proba(train)

		average_precision_tr,f1_tr,auc_tr = pr_curve(y ,y_pred_tr[:,1])
		average_precision_te,f1_te,auc_te = pr_curve(labels_te['used?'] ,y_pred_te[:,1])
	elif algo == 2: # XGBoost
		#pdb.set_trace()
		X = X.values
		y = y.values
		X_test = test.values
		y_te = y_te.values
		dtrain = xgb.DMatrix(X, label=y)
		dtest = xgb.DMatrix(X_test, label=y_te)
		param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
		num_round = 20
		bst = xgb.train(param, dtrain, num_round)
		# make prediction
		preds = bst.predict(dtest)
		average_precision_te,f1_te,auc_te = pr_curve(y_te ,preds)
	print average_precision_te,f1_te,auc_te
else:
	targets = ['used?','repurchase_15?','repurchase_30?','repurchase_60?','repurchase_90?']
	test_csv_file = './reactivation_data/predict.csv'
	test_info = read_test(test_csv_file)
	test = process_train(test_info,0,reactive_data_new,user_profiles)
	train = train_record_new
	test_pred = test_info.copy()
	for tth in range(0,len(targets)):
		X = train
		y = labels[targets[tth]]
		lr = linear_model.LogisticRegression(C=10)
		lr.fit(X, y)
		y_pred_te = lr.predict_proba(test)
		test_pred[targets[tth]] = pd.Series(y_pred_te[:,1], index=test_pred.index)
		y_pred_tr = lr.predict_proba(train)
		average_precision_tr,f1_tr,auc_tr = pr_curve(y ,y_pred_tr[:,1])
		print auc_tr
	test_pred.to_csv('result.csv',index=False)
	print 'Result saved!'
#pdb.set_trace()