import itertools
from pyspark import SparkConf, SparkContext


def line_to_friend_list(line):
    split = line.split()
    user_id = int(split[0])
    if len(split) == 1:
        friends = []
    else:
        friends = list(map(lambda x: int(x), split[1].split(',')))
    return user_id, friends


def friend_list_to_connection(friend_list):
    user_id, friends = friend_list

    connections = []

    if not friends:
        return [((user_id, user_id), -1)]

    for friend_id in friends:
        key = (user_id, friend_id)
        if user_id > friend_id:
            key = (friend_id, user_id)

        connections.append(
            (key, 0)
        )

    for friend_pair in itertools.combinations(friends, 2):
        friend_0, friend_1 = friend_pair

        key = (friend_0, friend_1)
        if friend_0 > friend_1:
            key = (friend_1, friend_0)
        connections.append(
            (key, 1)
        )

    return connections


def mutual_friend_count_to_recommendation(mutuals):
    connection, count = mutuals

    friend_0, friend_1 = connection

    recommendation_0 = (friend_0, (friend_1, count))
    recommendation_1 = (friend_1, (friend_0, count))

    return [recommendation_0, recommendation_1]


def recommendation_to_sorted_truncated(recommendation):
    user = recommendation[0]
    recommendations = list(recommendation[1])

    # Sort first by mutual friend count, then by user_id (for equal number of mutual friends between users)
    recommendations.sort(key=lambda x: (-x[1], x[0]))

    if recommendations[-1][0] == user:
        return user, []

    # Truncate to 10 elements and map to result lines
    return user, recommendations[:10]


def recommendation_to_string(recom):
    user, recommendations = recom

    if not recommendations:
        return str(user)

    return str(user) + '\t' + ','.join(list(map(lambda x: str(x[0]), recommendations)))

conf = SparkConf()
sc = SparkContext(conf=conf)
lines = sc.textFile('2.txt')
friend_ownership = lines.map(line_to_friend_list)
friend_edges = friend_ownership.flatMap(friend_list_to_connection)

mutual_friend_counts = friend_edges.groupByKey() \
    .filter(lambda edge: 0 not in edge[1]) \
    .map(lambda edge: (edge[0], sum(edge[1])))

recommendations = mutual_friend_counts.flatMap(mutual_friend_count_to_recommendation) \
    .groupByKey() \
    .map(recommendation_to_sorted_truncated) \
    .sortBy(lambda x: x[0]) \
    .map(recommendation_to_string)

recommendations.repartition(1).saveAsTextFile('result')
sc.stop()
