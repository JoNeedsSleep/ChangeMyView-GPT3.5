import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

attributes = ['formality', 'subjectivity', 'optimistic vs. cynical tone', 'extremity', 'lexical density','persuasiveness'] 

def violin_plot(df):
    palette = {"positive": "#5272F2", "negative": "#DAF1F9"}


    plt.figure(figsize=(20, 8))

    # Melt the DataFrame to have 'argument_type', 'variable', and 'value'
    melted_df = df.melt(id_vars=['argument_type'], value_vars=attributes, var_name='attribute', value_name='value')


    # Draw the violinplot with 'attribute' on x-axis and 'value' on y-axis
    # 'hue' differentiates between the argument types
    sns.violinplot(data=melted_df, x='attribute', y='value', hue='argument_type', linewidth=0, palette=palette, alpha=0.7, dodge=False, density_norm='width')

    #cleaner display
    sns.set_style("white")

    plt.title('Distributions of Semantic Classification by Positive/Negative')
    plt.legend(loc='upper left')

    #high res
    plt.savefig('attribute_violin.png', dpi=300)
    
    plt.show()
def corr_matrix(df):
    # Calculate Correlation Matrix
    correlation_matrix = df[attributes].corr()

    # Plot the Correlation Matrix
    plt.figure(figsize=(15, 9))
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues')
    plt.title('Correlation Matrix')
    #high res
    plt.savefig('correlation_matrix.png', dpi=300)
    
    plt.show()


def KDE_scatter_plot(df, y_prob):
    # Set the style of the visualization
    sns.set(style="white")

    # Create a dataframe for the KDE plot
    data_for_plot = pd.DataFrame({'Persuasiveness Score': df['persuasiveness'], 'Predicted Probability': y_prob})

    # Create a 2D KDE plot
    plt.figure(figsize=(15, 9))  # Set the size of the figure
    ax = sns.kdeplot(
        data=data_for_plot, 
        x="Persuasiveness Score", 
        y="Predicted Probability", 
        fill=True,  # Fill the density
        thresh=0,  # Include all contours
        levels=18,  # Number of contour levels, more levels will make the plot smoother
        cmap="Blues"  # Color map to use for the plot
    )

    sns.regplot(
        data=data_for_plot,
        x='Persuasiveness Score',
        y='Predicted Probability',
        scatter=False,  # We don't want to plot the scatter again
        color="purple",
        ci=99.9,
        line_kws={"linewidth": 2, "linestyle": "--"}
    )


    # Set titles and labels
    ax.set_title('KDE Plot + Regression Line of GPT-3.5 Persuasiveness rating and Linear Regression Model Predicted Probability', fontsize=16)
    ax.set_xlabel('Persuasiveness Score', fontsize=14)
    ax.set_ylabel('Predicted Probability', fontsize=14)


    plt.savefig('logreg_KDE_scatter.png', dpi=300)

    plt.show()
