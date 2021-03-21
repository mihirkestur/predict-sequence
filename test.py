t = int(input())
for i in range(1,1+t):
    R, C = input().split()
    array = []
    for i in range(R):        
        row = [] 
        for j in range(C):      
            row.append(int(input())) 
        array.append(row) 
    
    print("Case #{}: {}".format(i, answer))