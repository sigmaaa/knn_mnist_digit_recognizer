import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from dash import Dash, html, dcc, dash_table, Input, Output
from utility.knn import KNN

# Load digits and create preview image
digits = load_digits()
fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Training: {label}")

buf = io.BytesIO()
plt.tight_layout()
plt.savefig(buf, format="png")
plt.close(fig)
buf.seek(0)
encoded_image = base64.b64encode(buf.read()).decode("utf-8")
image_src = f"data:image/png;base64,{encoded_image}"

# Prepare dataset
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=False
)

# Precompute results for each (k, distance)
precomputed_results = {}
for k in [3, 5, 7]:
    for distance in [KNN.EUCLIDEAN, KNN.MANHATTAN]:
        model = KNN(k=k, distance=distance, x_train=X_train, y_train=y_train)
        y_pred = model.predict(X_test)

        # Classification result table
        table_data = [
            {
                "Index": i,
                "True Label": int(true),
                "Predicted Label": int(pred),
                "Match": "✅" if true == pred else "❌"
            }
            for i, (true, pred) in enumerate(zip(y_test, y_pred))
        ]

        # Example neighbor distances for first test sample
        test_point = X_test[0]
        # assumes method returns [(dist, label), ...]
        neighbors = model.get_neighbors(test_point)

        neighbor_info = [
            {"Label": label, "Distance": float(dist)}
            for dist, label in neighbors
        ]

        # Store all results
        accuracy = np.mean(y_pred == y_test)
        precomputed_results[(k, distance)] = {
            "data": table_data,
            "accuracy": round(accuracy, 4),
            "neighbors": neighbor_info
        }

# Create Dash app
app = Dash(__name__)
app.layout = html.Div([
    html.H1("KNN Dashboard"),
    html.H2("Digit Samples"),
    html.Img(src=image_src),

    html.Div([
        html.Label("Select distance metric:"),
        dcc.Dropdown(
            id="distance-dropdown",
            options=[
                {"label": "Euclidean", "value": KNN.EUCLIDEAN},
                {"label": "Manhattan", "value": KNN.MANHATTAN},
            ],
            value=KNN.EUCLIDEAN,
            clearable=False
        )
    ], style={'width': '30%', 'margin-bottom': '20px'}),

    html.Div([
        html.H2("Explore Neighbors for Test Samples"),
        dcc.Slider(
            id='index-slider',
            min=0,
            max=len(X_test) - 1,
            step=1,
            value=0,
            marks={0: '0', len(X_test) - 1: str(len(X_test) - 1)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Div(id='neighbor-table-output')
    ], style={'margin-top': '40px'}),
    dcc.Tabs(id="tabs-k", value='k-3', children=[
        dcc.Tab(label='k=3', value='k-3'),
        dcc.Tab(label='k=5', value='k-5'),
        dcc.Tab(label='k=7', value='k-7')
    ]),

    html.Div(id='tabs-content')
])


@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs-k', 'value'),
    Input('distance-dropdown', 'value')
)
def update_output(tab_value, distance):
    k_value = int(tab_value.split('-')[1])
    result = precomputed_results[(k_value, distance)]

    return html.Div([
        html.H3(
            f"KNN Results (k={k_value}, distance={distance}) - Accuracy: {result['accuracy']:.2%}"),
        dash_table.DataTable(
            data=result['data'],
            columns=[
                {"name": "Index", "id": "Index"},
                {"name": "True Label", "id": "True Label"},
                {"name": "Predicted Label", "id": "Predicted Label"},
                {"name": "Match", "id": "Match"},
            ],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            page_size=10,
            sort_action="native",
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Match} = "❌"'},
                    'backgroundColor': '#ffe6e6',
                },
                {
                    'if': {'filter_query': '{Match} = "✅"'},
                    'backgroundColor': '#e6ffe6',
                },
            ]
        )
    ])


@app.callback(
    Output('neighbor-table-output', 'children'),
    Input('tabs-k', 'value'),
    Input('distance-dropdown', 'value'),
    Input('index-slider', 'value')
)
def update_neighbors(tab_value, distance, test_idx):
    k_value = int(tab_value.split('-')[1])
    model = KNN(k=k_value, distance=distance, x_train=X_train, y_train=y_train)

    test_point = X_test[test_idx]
    # should return [(dist, label), ...]
    neighbors = model.get_neighbors(test_point)

    neighbor_info = [
        {"Label": label, "Distance": float(dist)}
        for dist, label in neighbors
    ]

    return html.Div([
        html.H3(f"Neighbors for Test Sample Index {test_idx}"),
        dash_table.DataTable(
            data=neighbor_info,
            columns=[
                {"name": "Label", "id": "Label"},
                {"name": "Distance", "id": "Distance"}
            ],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            page_size=10,
            sort_action="native"
        )
    ])


if __name__ == "__main__":
    app.run(debug=True)
