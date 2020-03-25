
#compare two files
import sys

file1_name = sys.argv[1]
file2_name = sys.argv[2]

file1 = open(file1_name,'r')
entire_file = file1.readlines()
numbers = entire_file[0].split(',')
print(len(numbers))
len_file1 = len(numbers)-1

file2 = open(file2_name,'r')
entire_file2 = file2.readlines()
numbers2 = entire_file2[0].split(',')
print(len(numbers2))
len_file2 = len(numbers2)-1

if len_file1<len_file2:
    len_min = len_file1
    extracted_numbers1 = numbers[0:len_min]
    extracted_numbers2 = numbers2[0:len_min]
    #print(len(extracted_numbers))
else:
    len_min = len_file2
    extracted_numbers1 = numbers[0:len_min]
    extracted_numbers2 = numbers2[0:len_min]

    
    
non_matching = 0
nums_compared = 0
for i in range(0,len_min):
    nums_compared = nums_compared+1
    if numbers[i] != numbers2[i]:
        non_matching = non_matching+1
        print ("Do not match %s and %s index: %d"%(numbers[i],numbers2[i],i))

print ("%d numbers do not match"%(non_matching))
print("%d numbers compared"%(nums_compared))