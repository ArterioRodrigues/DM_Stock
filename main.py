



def blackbox(arr):
  count = 1
  for i in arr:
    for j in arr:
      print(f'Got data => i: {i}, j: {j} \n count: {count}')
      count += 1


input=[] #empty
input2=[1, 2, 3] # 1 element #s1: 1, s2: 4, s3: 9
input3=[100, 1, 2, 3] #100,000 elements

blackbox(input2)
# blackbox(input2)
# blackbox(input3)

#[1, 2, 3]
#       i
#[1, 2, 3]
#       j

