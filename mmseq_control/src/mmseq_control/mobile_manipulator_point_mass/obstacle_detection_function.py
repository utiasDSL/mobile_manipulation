import casadi as ca

def object_detection_func(self, obstacles=[]):
    q = ca.MX.sym('q', 3)
    res = 0
    balls = self.generate_balls_constraints(q)
    for obstacle in obstacles:
        for ball in balls:
            res += ca.fmin(0, ca.norm_2(ball - ca.vertcat(obstacle.x, obstacle.y)) - obstacle.radius - self.ball_radius)
    
    # create a ca function
    return ca.Function('obstacle_detection', [q], [res])
