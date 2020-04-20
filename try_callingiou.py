import calculate_iou_2obj as cal_mul
acc_loss = cal_mul.cal_iou('log_org.txt', 'log_test.txt')
print(acc_loss)