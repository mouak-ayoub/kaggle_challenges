import pandas as pd

DRAW = -1
HOME_WINS = 1
AWAY_WINS = -2


def prepare_attribut_data(path_away: str, path_home: str, columns: list[str] = [],
                          use_player_data: bool = False, is_test: bool = False

                          ):
    '''

    :param path_away:
    :param path_home:
    :param columns: if columns is empty so take into account all columns
    :param is_test: test data doesn't have LEAGUE and TEAM_NAME columns
    :return:
    '''
    df_away = pd.read_csv(path_away)
    df_home = pd.read_csv(path_home)
    if columns:
        df_away = df_away.filter(items=columns)
        df_home = df_home.filter(items=columns)

    # If not test data and columns is all (so empty)
    if not is_test and not columns:
        team_col_to_drop = ['LEAGUE', 'TEAM_NAME']
        df_away = df_away.drop(team_col_to_drop, axis=1)
        df_home = df_home.drop(team_col_to_drop, axis=1)
    df_home.columns = [('HOME_' + str(col)) if col != 'ID' else 'ID' for col in df_home.columns]
    df_away.columns = [('AWAY_' + str(col)) if col != 'ID' else 'ID' for col in df_away.columns]
    df_joined = df_home.join(df_away.set_index('ID'), on='ID')
    if use_player_data:
        path_player_away = path_away.replace('team', 'player')
        path_player_home = path_home.replace('team', 'player')
        playes_col_to_drop = ['LEAGUE', 'TEAM_NAME', 'POSITION', 'PLAYER_NAME'] if not is_test else ['POSITION']

        df_player_away = pd.read_csv(path_player_away).drop(playes_col_to_drop, axis=1)
        df_player_home = pd.read_csv(path_player_home).drop(playes_col_to_drop, axis=1)

        df_player_away.columns = [('AWAY_' + str(col)) if col != 'ID' else 'ID' for col in df_player_away.columns]
        df_player_home.columns = [('HOME_' + str(col)) if col != 'ID' else 'ID' for col in df_player_home.columns]

        df_player_away = df_player_away.fillna(0.0)
        df_player_home = df_player_home.fillna(0.0)

        df_player_away_agg = df_player_away.groupby('ID', as_index=False).sum()
        df_player_home_agg = df_player_home.groupby('ID', as_index=False).sum()

        df_joined = (
                      df_joined.join(df_player_away_agg.set_index('ID'), on='ID')
                     .join(df_player_home_agg.set_index('ID'), on='ID')
        )
    df_joined = df_joined.fillna(0.0)
    df_final = df_joined.sort_values(by=['ID'])

    return df_final


def prepare_result_data(path_result):
    df_result = pd.read_csv(path_result)
    df_result.loc[df_result['DRAW'] == 1, 'result'] = DRAW
    df_result.loc[df_result['HOME_WINS'] == 1, 'result'] = HOME_WINS
    df_result.loc[df_result['AWAY_WINS'] == 1, 'result'] = AWAY_WINS
    df_result['result'] = df_result['result'].astype('int')
    df_result_final = df_result.sort_values(by=['ID']).iloc[:, [4]]

    return df_result_final


def convert_to_one_hot(yhat_predicted):
    y_pred_test = pd.DataFrame(yhat_predicted)
    y_pred_test.columns = ['prediction']
    y_pred_test.loc[y_pred_test['prediction'] == HOME_WINS, 'HOME_WINS'] = 1
    y_pred_test.loc[y_pred_test['prediction'] == DRAW, 'DRAW'] = 1
    y_pred_test.loc[y_pred_test['prediction'] == AWAY_WINS, 'AWAY_WINS'] = 1
    y_pred_test = y_pred_test.fillna(0)

    y_pred_test['HOME_WINS'] = y_pred_test['HOME_WINS'].astype('int')
    y_pred_test['DRAW'] = y_pred_test['DRAW'].astype('int')
    y_pred_test['AWAY_WINS'] = y_pred_test['AWAY_WINS'].astype('int')
    y_pred_test = y_pred_test.drop(['prediction'], axis=1)
    return y_pred_test
