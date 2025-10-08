from Custom_Logger import logger

class Executor():

    def perform_action(self, act):

        logger.info(    {'Component': "Executor" , "Action": "Performing Action" }  ) 
        # print('Inside Execute, performing action: ', act)
        logger.data( {"Action": act})
        

        # model switch takes place by changing model name in model.csv file .
        if (act == 1):
            # switch model to n
            logger.info(    {'Component': "Executor" , "Action": "Switching to model YOLOV5n" }  ) 
            f = open("model.csv", "w")
            f.write("yolov5n")
            f.close()

            logger.info(    {'Component': "Executor" , "Action": "Finished Action 1" }  ) 

        elif (act == 2):
            # switch model to s
            logger.info(    {'Component': "Executor" , "Action": "Switching to model YOLOV5s" }  ) 
            f = open("model.csv", "w")
            f.write("yolov5s")
            f.close()
            logger.info(    {'Component': "Executor" , "Action": "Finished Action 1" }  )

        elif (act == 3):
            # switch model to m
            logger.info(    {'Component': "Executor" , "Action": "Switching to model YOLOV5m" }  ) 
            f = open("model.csv", "w")
            f.write("yolov5m")
            f.close()
            logger.info(    {'Component': "Executor" , "Action": "Finished Action 1" }  )

        elif (act == 4):
            # switch model to l
            logger.info(    {'Component': "Executor" , "Action": "Switching to model YOLOV5l" }  ) 
            f = open("model.csv", "w")
            f.write("yolov5l")
            f.close()
            logger.info(    {'Component': "Executor" , "Action": "Finished Action 1" }  )
        elif (act == 5):
            logger.info(    {'Component': "Executor" , "Action": "Switching to model YOLOV5x" }  ) 
            # switch model to l
            f = open("model.csv", "w")
            f.write("yolov5x")
            logger.info(    {'Component': "Executor" , "Action": "Finished Action 1" }  )

        print("Adaptation completed.")
