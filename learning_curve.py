import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, train_sizes = np.linspace(0.1, 1, 5)):
    """
    画出data在某个模型上的learning curve
    参数解释：
    ————
    estimoar: 模型
    title：dataframe 的标题
    X：输入的feature，numpy类型
    y: 输入的target vector
    ylim: tuple格式的(ymin, ymax)， 设定图像纵坐标的最低点和最高点
    cv: 做cv的时候，数据分成的份数
    """
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5,n_jobs = 1, train_sizes=train_sizes) 
    
    
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.std(test_scores, axis = 1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                        train_scores_mean + train_scores_std, alpha = 0.1, color = 'r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha = 0.1, color = 'g')
    
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color = 'r', label = 'Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color = 'g', label = 'Cross-validation score')
    
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc = 'best')
    plt.grid('on')
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()