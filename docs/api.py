"""API for python FCM package.
FCM data is encapsulated on an FCM objects.
FCM object methods should be chainable - see chain.py

e.g. 

fcm = FCM()
fcm.read('fcsfile')
fcm = fcm.select_cols([1,2,3,5]).compensate(fcm, spillover).logicle(m,r,T)

"""

def compensate(fcm, spillover):
    """Perform compensation using spillover matrix."""
    pass

