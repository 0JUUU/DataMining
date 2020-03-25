import numpy 

oneDim = numpy.array([1,0,2,3,4,5])
print(oneDim)
print("#Dimensions =", oneDim.ndim)
print("Dimension =", oneDim.shape)
print("Size =", oneDim.size)
print("Array type =",oneDim.dtype)

twoDim = numpy.array([[1,2],[3,4],[5,6],[7,8]])
print(twoDim)
print("#Dimensions =", twoDim.ndim)
print("Dimension =", twoDim.shape)
print("Size =", twoDim.size)
print("Array type =",twoDim.dtype)

arrFromTuple = numpy.array([(1,'a',3.0),(2,'b',3.5)])
print(arrFromTuple)
print("#Dimensions =", arrFromTuple.ndim)
print("Dimension =", arrFromTuple.shape)
print("Size =", arrFromTuple.size)