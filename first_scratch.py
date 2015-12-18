#!/usr/bin/python
# This script is to check for parameter searching. English and gender.

import sys
import getopt
import bleach
import xml.etree.ElementTree as ET
import os
import re
import csv

reload(sys)
sys.setdefaultencoding("ISO-8859-1")


def dirExists(inputDir):
	if os.path.exists(inputDir):
		return True
	elif os.access(os.path.dirname(inputDir), os.W_OK):
		print "Cannot access the directory. Check for privileges."
		return False
	else:
		print "Directory does not exist."
		return False

def absoluteFilePaths(directory):
	allPaths = []
	for dirpath,_,filenames in os.walk(directory):
		for f in filenames:
			onePath = os.path.abspath(os.path.join(dirpath, f))
			allPaths.append(onePath)
# 			yield os.path.abspath(os.path.join(dirpath, f))
	return allPaths


def getTruthTextFiles(allPaths):
	return [f for f in allPaths if isTruthTextFile(f)]

def getAllFilenamesWithAbsPath(inputDir):
	if dirExists(inputDir):
		allPaths = absoluteFilePaths(inputDir)
		return allPaths
	else:
		sys.exit()

def isTruthTextFile(f):
	return 'truth.txt' in f

def getRelevantDirectories(argv):
   inputDir = ''
   outputDir = ''
   modelDir = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print './myTrainingSoftware.py -i <inputdirectory> -o <outputdirectory>'
      print 'The input directory should contain all the training files. \nThe output directory will be where the models are stored.'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print './myTrainingSoftware.py -i <inputdirectory> -o <outputdirectory>'
         print 'The input directory should contain all the training files. \nThe output directory will be where the models are stored.'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputDir = arg
      elif opt in ("-o", "--ofile"):
         outputDir = arg   
   return inputDir, outputDir


def tsv_writer(data, path):
    """
    Write data to a TSV file path
    """
    with open(path, "a") as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        writer.writerow(data)

def writeOneSummary(outputFilename, f, absPathToInputDir):
	''' Writes a tsv file that summarizes the things needed for the task.
	'''
	data = ["filename", "gender", \
			"age", "extroverted", \
			"stable", "agreeable", \
			"open", "conscientious", \
                        "text", "num_tweets_per_user", \
                        "total_length_of_tweet", "avg_tweet_length_per_user"]
	tsv_writer(data, outputFilename)

	gender = {'M': 0, 'F':1}
	ageGroup = {'18-24': 0, \
				'25-34': 1, \
				'35-49': 2, \
				'50-XX': 3, \
				'50-64': 3, \
				'XX-XX': None}
	file1 = open(f, 'r')
	
	for line in file1:
		#a = line.strip().split(":::")
		a = line.split(":::")
		fileName 		  = absPathToInputDir + "/" + a[0] + ".xml"
# 		print fileName
                #import pdb
                #pdb.set_trace()
		thisGender 	 	  = gender[a[1]]
		thisAgeGroup 	  = ageGroup[a[2]]
		thisExtroverted   = float(a[3])
		thisStable 		  = float(a[4])
		thisAgreeable	  = float(a[5])
		thisOpen		  = float(a[6])
		thisConscientious = float(a[7])
		
# 		print "%s %d %d %f %f %f %f %f" % (fileName, thisGender, thisAgeGroup, thisExtroverted, thisStable, thisAgreeable, thisOpen, thisConscientious)
		
		try:
			tree = ET.parse(fileName)
			print "Filename: %s SUCCESS!" % fileName
		
		except:
			e = sys.exc_info()[0]
			print "Filename: %s Error: %s" % (fileName, e)
		else:
			allDocs = tree.getroot().findall("document")
 			allText = ""

# min and max number of tweets per user
# average tweet length per user (max and min)
# total average tweet length
                        num_tweets_per_user = len(allDocs)
                        num_chars_in_one_tweet = 0

                        #all_clean_text = [bleach.clean(each_doc.text, tags=[], strip=True).encode('utf-8') for each_doc in allDocs]
                        #len_each_item_in_all_clean_text = [len(a) for a in all_clean_text] 
                        #allText = " ".join(all_clean_text)
                        #total_length_of_tweet = len(allText)
                       
                       
                        for doc in allDocs:
				clean = bleach.clean(doc.text, tags=[], strip=True)
                                num_chars_in_one_tweet = num_chars_in_one_tweet + len(clean)
 				
                                allText = allText + clean
	 	    	allText = allText.encode('utf-8')
			#	clean = clean.encode('utf-8')								
                        num_chars_in_one_tweet = len(allText)
                        
                        total_length_of_tweet = num_chars_in_one_tweet
                        
                        avg_tweet_length_per_user = total_length_of_tweet/float(num_tweets_per_user)
			data = [fileName, thisGender, thisAgeGroup, 
                                thisExtroverted, thisStable, 
                                thisAgreeable, thisOpen, thisConscientious,
                                allText, num_tweets_per_user, total_length_of_tweet,
                                avg_tweet_length_per_user]
			tsv_writer(data, outputFilename)


def main(argv):
	inputDir, outputDir = getRelevantDirectories(argv)
	allPaths = getAllFilenamesWithAbsPath(inputDir)
	print "ALL PATHS\n\n"
	print allPaths
	print "\n\n"
	allTruthText = getTruthTextFiles(allPaths)
	allTruthText = [f for f in allTruthText if "english" in f]
	print "ALL TRUTH TEXT\n\n"
	print allTruthText
	print "\n\n"
	absPathToInputDir = allTruthText[0].strip().split('/')
	absPathToInputDir = '/'.join(absPathToInputDir[0:-1])
	print "ABS PATH TO INPUT DIR\n\n"
	print absPathToInputDir
	models = {}
# # 	tasks = ["gender", "age", "extroverted", "stable", "agreeable", "open", "conscientious"]
	tasks = ["gender"]
	for f in allTruthText:
		a = f.strip().split("/")
		outputFilename = './summary-' + a[-1]
		print "OUTPUT FILENAME"
		print outputFilename

		writeOneSummary(outputFilename, f, absPathToInputDir)
# 		descriptors = getDescriptorsForOne(outputFilename)
# 		model_for_one = {}
# 		for task in tasks:
# 			y = getTarget(outputFilename, task)
# 			model_for_one[task] = trainOne(descriptors, y)
# 		models[f] = model_for_one
# 	models = trainAll(inputDir)
# 	writeModels(models, outputDir)

if __name__ == "__main__":
   main(sys.argv[1:])
