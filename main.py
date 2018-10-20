print("<------------------------------Political Tweet Analyser------------------------------>")
print("Have an idea of the political scenario of our country by checking out the sentiments\nof the latest 1000 tweets concerning the three major political parties \n(BJP,Congress,AAP)")
input_ = input("Press 1 for BJP\nPress 2 for Congress\nPress 3 for AAP\n")
if(input_=="1"):
    exec(compile(open('bjp_data.py').read(), 'bjp_data.py', 'exec'))
elif(input_=="2"):
    exec(compile(open('inc_data.py').read(), 'inc_data.py', 'exec'))
elif(input_=="3"):
    exec(compile(open('aap_data.py').read(), 'aap_data.py', 'exec'))
else:
    print("Invalid Input")


