# this is a python file to show that we can compress any text-based file!

x = 'Hello World'
print(x)

d = {}
for c in x:
    d[c] = d.get(c, 0)
    d[c] += 1

for k, v in d.items():
    print(k, v)

quit()

