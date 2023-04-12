class A:
    def __init__(self) -> None:
        print("Constructore of A")
        self.method()
    
    def method(self):
        print("Method of A")
        
class B(A):
    def __init__(self) -> None:
        print("B")
        self.method()
        
    # def method(self):
    #     print("Method of B")
        
b = B()