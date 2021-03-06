# -*- coding: utf-8 -*-
"""Manim For DQN

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/BeeGass/Deep-Q-Learning/blob/main/Manim_For_DQN.ipynb
"""

!sudo apt update
!sudo apt install libcairo2-dev ffmpeg texlive texlive-latex-extra texlive-fonts-extra texlive-latex-recommended texlive-science tipa libpango1.0-dev
!pip install git+https://github.com/ManimCommunity/manim.git
!pip install IPython --upgrade

from manim import *
import numpy as np
import itertools as it
import random

# Commented out IPython magic to ensure Python compatibility.
# %%manim --disable_caching -qp -v WARNING NeuralNetwork
# 
# class RenderTableOfContents(Scene):
#     def construct(self):
#         config.background_color=BLACK
#         text = VGroup()
#         text.add(Text("DQN").scale(1))
#         text.add(Text("Deep Q-Network").scale(1))
#         text.add(Text("Q-Learning").scale(1))
#         text.add(Text("Q").scale(1))
# 
#         #load Q-learning
#         self.play(Write(text[3]))
#         self.wait()
#         self.play(Transform(text[3], text[2]))
#         self.wait()
#         text[2].to_edge(UP, buff=1)
#         text[2].to_edge(LEFT)
#         self.play(FadeTransform(text[3], text[2]))
# 
#         #load DQN
#         q = Text("Q").scale(1)
#         self.play(Write(q))
#         self.wait()
#         self.play(Transform(q, text[0]))
#         self.wait()
#         self.play(Transform(q, text[1]))
#         self.wait()
#         text[1].next_to(text[2], direction=(DOWN*1.5), aligned_edge=LEFT)
#         self.play(FadeTransform(q, text[1]))
#         
#         #load progress on DQN thus far
#         results = Text("The Results").scale(1)
#         self.play(Write(results))
#         self.wait()
#         rc = results.copy()
#         rc.next_to(text[1], direction=(DOWN*1.5), aligned_edge=LEFT)
#         self.play(FadeTransform(results, rc))
#         self.wait()
# 
# class RenderInitReplayBuffer(Scene):
#     def construct(self):
#         buffer_title = Tex("Initialize replay memory D to capacity N")
#         buffer_title.shift(2*UP)
#         buffer_title.set_color(YELLOW_B)
#         self.play(Write(buffer_title))
#         rects = VGroup()
#         num_rects = 25
#         for item in range(num_rects):
#             rect = Rectangle(height=0.5, width=0.5,color=BLUE)
#             rects.add(rect)
#         rects.arrange(buff=0)
#         self.play(ShowCreation(rects, run_time=4))
# 
# class RenderTransitionIntoReplayBuffer(ZoomedScene):
#     def construct(self):
#         buffer_title = Tex("Store Transition")
#         buffer_title.shift(2*UP)
#         buffer_title.set_color(YELLOW_B)
#         self.play(Write(buffer_title))
#         rects = VGroup()
#         num_rects = 25
#         zoomed_in_v_group = VGroup()
#         zoomed_in_v_group.add(Text("Current State").scale(0.05))
#         zoomed_in_v_group.add(Text("Action").scale(0.05))
#         zoomed_in_v_group.add(Text("Reward").scale(0.05))
#         zoomed_in_v_group.add(Text("Next State").scale(0.05))
# 
#         zoomed_in_v_group[1].next_to(zoomed_in_v_group[0], direction=(DOWN*0.025), aligned_edge=LEFT)
#         zoomed_in_v_group[2].next_to(zoomed_in_v_group[1], direction=(DOWN*0.025), aligned_edge=LEFT)
#         zoomed_in_v_group[3].next_to(zoomed_in_v_group[2], direction=(DOWN*0.025), aligned_edge=LEFT)
# 
#         for item in range(num_rects):
#             rect = Rectangle(height=0.5, width=0.5,color=BLUE)
#             if item == 24:
#                 arrow_wb = Arrow(np.array([6.0, 0.2, 0]), np.array([6.0, 0.1, 0]), buff=0)
#                 arrow_wb.set_color(RED)
#                 trans = MathTex("(\Phi_{t}, a_{t}, r_{t}, \Phi_{t+1})").scale(0.5)
#                 trans.next_to(arrow_wb, UP)
#                 trans_brace = BraceLabel(obj=trans, text="Transition", brace_direction=UP)
#                 self.play(ShowCreation(arrow_wb))
#                 self.play(ShowCreation(trans))
#                 self.play(Write(trans_brace))
#                 rect.add(zoomed_in_v_group)
#             rects.add(rect)
#         rects.arrange(buff=0)
#         self.play(ShowCreation(rects, run_time=4))
#         self.activate_zooming(animate=True)
#         self.play(self.zoomed_camera.frame.animate.shift(6 * RIGHT))
#         self.wait(2)
# 
# class RenderEvictTransitionFromReplayBuffer(ZoomedScene):
#     def construct(self):
#         buffer_title = Tex("Evict Transition")
#         buffer_title.shift(2*UP)
#         buffer_title.set_color(YELLOW_B)
#         self.play(Write(buffer_title))
#         rects = VGroup()
#         num_rects = 25
# 
#         for item in range(num_rects):
#             rect = Rectangle(height=0.5, width=0.5, color=BLUE)
#             arrow = Arrow(np.array([-6.0, 0.2, 0]), np.array([-6.0, 0.1, 0]), buff=0)
#             arrow.set_color(RED)
#             trans = MathTex("(\Phi_{t}, a_{t}, r_{t}, \Phi_{t+1})").scale(0.5)
#             trans.next_to(arrow, UP)
#             zoomed_in_v_group = VGroup()
#             zoomed_in_v_group.add(Text("Current State").scale(0.05))
#             zoomed_in_v_group.add(Text("Action").scale(0.05))
#             zoomed_in_v_group.add(Text("Reward").scale(0.05))
#             zoomed_in_v_group.add(Text("Next State").scale(0.05))
# 
#             zoomed_in_v_group[1].next_to(zoomed_in_v_group[0], direction=(DOWN*0.025), aligned_edge=LEFT)
#             zoomed_in_v_group[2].next_to(zoomed_in_v_group[1], direction=(DOWN*0.025), aligned_edge=LEFT)
#             zoomed_in_v_group[3].next_to(zoomed_in_v_group[2], direction=(DOWN*0.025), aligned_edge=LEFT)
#             if item == 24:
#                 self.play(ShowCreation(arrow))
#                 self.play(ShowCreation(trans))
#             rect.add(zoomed_in_v_group)
#             rects.add(rect)
#         rects.arrange(buff=0)
#         self.play(ShowCreation(rects, run_time=4))
#         self.activate_zooming(animate=True)
#         self.play(self.zoomed_camera.frame.animate.shift(6 * LEFT))
#         self.play(Uncreate(rects[0]))
#         self.wait(2)
# 
# class RenderDecision(Scene):
#     def construct(self):
#         circle_A = Circle(radius=1, color=GREEN)
#         circle_A.add(MathTex("\epsilon < [0, 1]").scale(0.5))
#         circle_A.move_to(2.75 * UP)
# 
#         circle_B = Circle(radius=1, color=ORANGE)
#         circle_B.add(Tex("Random Action").scale(0.5))
#         circle_B.move_to(DOWN+(LEFT*1.5))
# 
#         circle_C = Circle(radius=1, color=ORANGE)
#         circle_C.add(Tex("Chosen Action").scale(0.5))
#         circle_C.move_to(DOWN+(RIGHT*1.5))
# 
#         arrow_A = Arrow(np.array([0.5, 1.8, 0]), np.array([1.5, 0.1, 0]), buff=0)
#         arrow_B = Arrow(np.array([-0.5, 1.8, 0]), np.array([-1.5, 0.1, 0]), buff=0)
# 
#         self.play(ShowCreation(circle_A))
#         self.play(ShowCreation(circle_B))
#         self.play(ShowCreation(circle_C))
#         self.play(ShowCreation(arrow_A))
#         self.play(ShowCreation(arrow_B))
# 
# 
# class RenderQLearningTable(Scene):
#     def construct(self):
#         config.background_color=BLACK
#         rects = VGroup()
#         num_rects = 64
#         range_range = range(num_rects)
#         a_terms = range_range[1:8]
#         s_terms = range(0, num_rects, 8)[1:]
# 
#         for item in range_range:
#             rect = Square(side_length=0.7, color=WHITE)
#             if item in a_terms:
#                 label = MathTex(f"a_{item}",color=BLUE_D)
#                 rect.add(label)
#             elif item in s_terms:
#                 label = MathTex(f"s_{round(item/8)}", color=RED)
#                 rect.add(label)
# 
#             if item == 9:
#                 rect.add(MathTex(f"0.7",color=YELLOW).scale(1))
#             elif item == 18:
#                 rect.add(MathTex(f"-1",color=YELLOW).scale(1))
# 
#             rects.add(rect)
# 
#         rects.arrange_in_grid(buff=0)
#         l1 = Tex("States", color=RED).scale(0.6).next_to(rects, LEFT)
#         l2 = Tex("Actions", color=BLUE_D).scale(0.6).next_to(rects, UP)
#         l3 = Tex("\\underline{Q-Table}").scale(0.8).next_to(l2, UP)
# 
#         rects.add(l1, l2, l3)
#         rects.center()
#         print(rects.get_x())
#         rects.shift(LEFT*0.5)
#         self.play(LaggedStart(*[Write(x, run_time=0.75) for x in rects], lag_ratio=0.04))
#         self.play(ApplyWave(rects))
# 
# class CheckerBoard(Scene):
#     def construct(self):
#         config.background_color=BLACK
#         colour_scheme = [DARK_GREY, WHITE]
#         rects = VGroup()
#         num_rects_row = 8
#         num_rects_col = 8
# 
#         for row in range(num_rects_row):
#             for col in range(num_rects_col): 
#                 if row % 2 == 0:
#                     r_color = colour_scheme[(col + 1) % len(colour_scheme)]
#                 else:
#                     r_color = colour_scheme[col % len(colour_scheme)]
# 
#                 rect = Square(side_length=0.9, fill_color=r_color, color=r_color, fill_opacity=1)
# 
#                 if row < 2 or row > 5:
#                     if row == 0 and col % 2 == 1:
#                         rect.add((Circle().set_stroke(color=WHITE, width=10)).scale(0.2))
#                     elif row == 1 and col % 2 == 0:
#                         rect.add((Circle().set_stroke(color=WHITE, width=10)).scale(0.2))
#                     elif row == 6 and col % 2 == 1:
#                         rect.add((Circle().set_stroke(color=BLACK, width=10)).scale(0.2))
#                     elif row == 7 and col % 2 == 0:
#                         rect.add((Circle().set_stroke(color=BLACK, width=10)).scale(0.2))
# 
#                 x1 = 2
#                 y1 = 5
# 
#                 x2 = 5
#                 y2 = 4
# 
#                 x3 = 5
#                 y3 = 2
# 
#                 x4 = 2
#                 y4 = 3
#                 if row == x1 and col == y1:
#                     rect.add(MathTex("s_{1_{2}}").scale(1))
#                 elif row == x2 and col == y2:
#                     rect.add(MathTex("s_{1_{1}}").scale(1))
#                 elif row == x3 and col == y3:
#                     rect.add(MathTex("s_{2_{3}}").scale(1))
#                 elif row == x4 and col == y4:
#                     rect.add(MathTex("s_{2_{4}}").scale(1))
# 
#                 rects.add(rect)
#         rects.arrange_in_grid(buff=0)
#         self.add(rects)
#         self.play(LaggedStart(*[DrawBorderThenFill(x, run_time=0.35) for x in rects], lag_ratio=0.05))
#         arrow_wa = Arrow(np.array([-2, -2, 0]), np.array([-1.5, -1.5, 0]), buff=0)
#         arrow_wa.set_color(RED)
# 
#         arrow_wb = Arrow(np.array([-0.2, -2.0, 0]), np.array([0.3, -1.5, 0]), buff=0)
#         arrow_wb.set_color(RED)
# 
#         arrow_ba = Arrow(np.array([2, 2, 0]), np.array([1.5, 1.5, 0]), buff=0)
#         arrow_ba.set_color(RED)
# 
#         arrow_bb = Arrow(np.array([0.2, 2.0, 0]), np.array([-0.3, 1.5, 0]), buff=0)
#         arrow_bb.set_color(RED)
# 
# 
#         self.play(ShowCreation(arrow_wa))
#         self.play(ShowCreation(arrow_wb))
#         self.play(ShowCreation(arrow_ba))
#         self.play(ShowCreation(arrow_bb))
#         self.wait(3)
# 
# class TicTacToe(Scene):
#     def construct(self):
#         config.background_color=BLACK
#         colour_scheme = [WHITE]
#         rects = VGroup()
#         num_rects_row = 3
#         num_rects_col = 3
# 
#         for row in range(num_rects_row):
#             for col in range(num_rects_col): 
#                 rect = Square(side_length=2, fill_color=WHITE, color=BLACK, fill_opacity=1)
#                 
#                 if row == 0 and col == 2:
#                     circle = Circle()
#                     circle.set_stroke(color=BLUE, width=5).scale(0.7)
#                     circle.add(MathTex("s_{1_{1}}", color = BLACK).scale(0.5))
#                     rect.add(circle)
# 
#                 elif row == 1 and col == 1:
#                     circle = Circle()
#                     circle.set_stroke(color=BLUE, width=5).scale(0.7)
#                     circle.add(MathTex("s_{2_{3}}", color = BLACK).scale(0.5))
#                     rect.add(circle)
# 
#                 elif row == 1 and col == 0:
#                     triangle = Triangle()
#                     triangle.set_stroke(color=RED, width=5).scale(0.7)
#                     triangle.add(MathTex("s_{1_{2}}", color = BLACK).scale(0.5))
#                     rect.add(triangle)
# 
#                 elif row == 2 and col == 2:
#                     sqaure = Square()
#                     sqaure.set_stroke(color=YELLOW, width=15).scale(0.7)
#                     sqaure.add(MathTex("?s_{2_{2}}", color = BLACK).scale(0.5))
#                     rect.add(sqaure)
# 
# 
#                 rects.add(rect)
#         rects.arrange_in_grid(buff=0)
#         self.add(rects)
#         self.play(LaggedStart(*[DrawBorderThenFill(x, run_time=0.35) for x in rects], lag_ratio=0.05))
#         self.wait(5)
# 
# class DQNTicTacToe(Scene):
#     def construct(self):
#         config.background_color=BLACK
#         colour_scheme = [WHITE]
#         rects = VGroup()
#         num_rects_row = 3
#         num_rects_col = 3
# 
#         for row in range(num_rects_row):
#             for col in range(num_rects_col): 
#                 rect = Square(side_length=2, fill_color=WHITE, color=BLACK, fill_opacity=1)
#                 
#                 if row == 0 and col == 2:
#                     circle = Circle()
#                     circle.set_stroke(color=BLUE, width=5).scale(0.7)
#                     circle.add(MathTex("s_{1_{1}}", color = BLACK).scale(0.5))
#                     rect.add(circle)
# 
#                 elif row == 1 and col == 1:
#                     circle = Circle()
#                     circle.set_stroke(color=BLUE, width=5).scale(0.7)
#                     circle.add(MathTex("s_{2_{3}}", color = BLACK).scale(0.5))
#                     rect.add(circle)
# 
#                 elif row == 1 and col == 0:
#                     triangle = Triangle()
#                     triangle.set_stroke(color=RED, width=5).scale(0.7)
#                     triangle.add(MathTex("s_{1_{2}}", color = BLACK).scale(0.5))
#                     rect.add(triangle)
# 
#                 elif row == 2 and col == 2:
#                     sqaure = Square()
#                     sqaure.set_stroke(color=YELLOW, width=15).scale(0.7)
#                     sqaure.add(MathTex("?s_{2_{2}}", color = BLACK).scale(0.5))
#                     rect.add(sqaure)
# 
#                 elif (row == 0 and col == 0) or (row == 0 and col == 1) or (row == 1 and col == 2)  or (row == 2 and col == 0) or (row == 2 and col == 1):
#                     sqaure = Square()
#                     sqaure.set_stroke(color=YELLOW, width=15).scale(0.7)
#                     sqaure.add(MathTex("?s_{2_{2}}", color = BLACK).scale(0.5))
#                     rect.add(sqaure)
# 
# 
#                 rects.add(rect)
#         rects.arrange_in_grid(buff=0)
#         self.add(rects)
#         self.play(LaggedStart(*[DrawBorderThenFill(x, run_time=0.35) for x in rects], lag_ratio=0.05))
#         self.wait(5)
# 
# class RenderDQNPsuedo(Scene):
#     def construct(self):
#         config.background_color=BLACK
#         dqn = Tex("DQN").scale(3)
#         deep_q_network = Tex("Deep Q-Network").scale(2)
#         small_deep_q_network = Tex("Deep Q-Network").scale(1)
#         self.play(Write(dqn))
#         self.wait()
#         self.play(Transform(dqn, deep_q_network))
#         self.wait()
#         self.play(FadeTransform(dqn, small_deep_q_network.to_edge(UP))) 
#         self.wait
# 
#         first_for_loop = Tex(r'\textbf{for} ', r'episode', r' = 1,', r' M ', r'\textbf{do}').scale(1)
#         first_for_loop[0].set_color(BLUE)
#         first_for_loop[1].set_color(GREEN)
#         first_for_loop[3].set_color(GREEN)
#         first_for_loop[4].set_color(BLUE)
# 
#         second_for_loop = Tex(r'\textbf{for},', r' t', r'= 1,', r' T', r' \textbf{do}').scale(1)
#         second_for_loop[0].set_color(BLUE)
#         second_for_loop[1].set_color(GREEN)
#         second_for_loop[3].set_color(GREEN)
#         second_for_loop[4].set_color(BLUE)
# 
#         end_for = Tex(r'\textbf{end for}').scale(1)
#         end_for[0].set_color(GREEN)
# 
#         end_for_b = Tex(r'\textbf{end for}').scale(1)
#         end_for_b[0].set_color(GREEN)
# 
#         pl = VGroup()
#         pl.add(Tex(r'Initialize replay memory D to capacity N').scale(1)) #1 
#         pl.add(Tex(r'Initialize action-value function Q with random weights').scale(1)) #2
#         pl.add(first_for_loop) #3
#         pl.add(Tex(r"Initialise sequence $s_{1} =  \big\{x_{1}\big\}$ and preprocessed sequenced $\Phi_{1} =  \Phi(s_{1})$").scale(1)) #4 
#         pl.add(second_for_loop) #5
#         pl.add(Tex(r"With probability $\epsilon$ select a random action $a_{t}$").scale(1)) #6 
#         pl.add(Tex(r'Otherwise select $a_{t} = max_{a} Q^{*}(\Phi(s_{t}), a; \Theta)$').scale(1)) #7
#         pl.add(Tex(r'Execute action $a_{t}$ in emulator and observe reward $r_{t}$ and image $x_{t+1}$').scale(1)) #8 
#         pl.add(Tex(r'Set $s_{t+1}$ and preprocess $\Phi_{t+1} = \Phi(s_{t+1})$').scale(1)) #9
#         pl.add(Tex(r'Store transition $(\Phi_{t}, a_{t}, r_{t}, \Phi_{t+1})$ in D').scale(1)) #10
#         pl.add(Tex(r'Sample random minibatch of transitions $(\Phi_{t}, a_{t}, r_{t}, \Phi_{j+1})$ from D').scale(1)) #11
#         pl.add(MathTex(r"y_{j} = \begin{cases} r_{j}, & \text{for terminal $\Phi_{j+1}$}\\ r_{j} + \gamma max_{a^{'}} Q(\Phi_{j+1}, a^{'}; \Theta), & \text{for non-terminal $\Phi_{j+1}$} \end{cases}").scale(1)) #12
#         pl.add(Tex(r'Perform a gradient descent step on $(y_{j} - Q(\Phi_{j}, a_{j}; \Theta))^{2}$ according to equation 3').scale(1)) #13
#         pl.add(end_for) #14
#         pl.add(end_for_b) #15
# 
#         for i in range(len(pl)):
#             if i <= len(pl) and i >= 1:
#                 if i == 3 or i == 4:
#                     if i == 3:
#                         pl[i].next_to(pl[i-1], direction=DOWN, aligned_edge=LEFT)
#                         pl[i].shift(RIGHT)
#                     else:
#                         pl[i].next_to(pl[i-1], direction=DOWN, aligned_edge=LEFT)
#                 elif i >= 5 and i <= 12:
#                     if i == 5:
#                         pl[i].next_to(pl[i-1], direction=DOWN, aligned_edge=LEFT)
#                         pl[i].shift(RIGHT)
#                     else: 
#                         pl[i].next_to(pl[i-1], direction=DOWN, aligned_edge=LEFT)
#                 else:
#                     if i >= 0 and i <=2:
#                         pl[i].next_to(pl[i-1], direction=DOWN, aligned_edge=LEFT)
#                     elif i >= 13:
#                         pl[i].next_to(pl[i-1], direction=DOWN, aligned_edge=LEFT)
#                         pl[i].shift(LEFT)
# 
#         pl.scale_in_place(0.5)
#         pl.to_edge(UP, buff=1)
#         pl.to_edge(LEFT, buff=0.5)
#         pl.set_opacity(0.5)
#         pl[0].set_opacity(1)
#         self.play(Write(pl))
#         for i in range(len(pl)):
#             if i >= 1:
#                 self.add(pl[i-1].set_opacity(0.5))
#             self.play(Circumscribe(pl[i].set_opacity(1)))
#             #self.add(pl[i].set_opacity(1))
#             self.wait()
# 
# class RenderQValueExplanation(Scene):
#     def construct(self):
#         config.background_color=BLACK
#         circle_A = Circle(radius=0.5, color=GREEN)
#         circle_A.add(MathTex("s_{1}").scale(1))
#         circle_A.move_to((2.75 * UP) + RIGHT)
# 
#         circle_B = Circle(radius=0.5, color=ORANGE)
#         circle_B.add(MathTex("a_{...}").scale(1))
#         circle_B.move_to((UP * 0.25 ) + (LEFT * 0.25))
# 
#         circle_C = Circle(radius=0.5, color=ORANGE)
#         circle_C.add(MathTex("a_{1}").scale(1))
#         circle_C.move_to((UP * 0.25) + (RIGHT * 2.25))
# 
#         circle_CA = Circle(radius=0.5, color=ORANGE)
#         circle_CA.add(MathTex("s_{...}").scale(1))
#         circle_CA.move_to((DOWN * 2) + (RIGHT * 1.0))
# 
#         circle_CB = Circle(radius=0.5, color=ORANGE)
#         circle_CB.add(MathTex("s_{2}").scale(1))
#         circle_CB.move_to((DOWN * 2) + (RIGHT * 3.5))
# 
#         arrow_A = Arrow(np.array([1.3, 2.3, 0]), np.array([2.2, 0.75, 0]), buff=0)
#         arrow_B = Arrow(np.array([0.7, 2.3, 0]), np.array([-0.2, 0.75, 0]), buff=0)
# 
#         arrow_AA = Arrow(np.array([2.5, -0.2, 0]), np.array([3.5, -1.50, 0]), buff=0)
#         arrow_AB = Arrow(np.array([2.0, -0.2, 0]), np.array([1.0, -1.50, 0]), buff=0)
# 
#         reward = MathTex("r_{1}").scale(1)
# 
#         reward.move_to((UP * 1.75) + (RIGHT * 2.25))
# 
#         self.play(ShowCreation(circle_A))
#         self.play(ShowCreation(circle_B))
#         self.play(ShowCreation(circle_C))
#         self.play(ShowCreation(circle_CA))
#         self.play(ShowCreation(circle_CB))
#         self.play(ShowCreation(arrow_A))
#         self.play(ShowCreation(reward))
#         self.play(ShowCreation(arrow_B))
#         self.play(ShowCreation(arrow_AA))
#         self.play(ShowCreation(arrow_AB))
# 
# # class RenderPreprocessor(Scene):
# #     def constructor(self):
# #         black_box = Rectangle().size(2)
# 
# ## ripped from - https://www.youtube.com/watch?v=RLCqjCAbd5E ##
# class NeuralNetwork(Scene):
#     arguments = {
#         "network_size": 1,
#         "network_position": ORIGIN,
#         "layer_sizes": [3, 5, 5, 5, 3],
#         "layer_buff": LARGE_BUFF,
#         "neuron_radius": 0.15,
#         "neuron_color": LIGHT_GREY,
#         "neuron_width": 3,
#         "neuron_fill_color": MAROON_E,
#         "neuron_fill_opacity": 1,
#         "neuron_buff": MED_SMALL_BUFF,
#         "edge_color": YELLOW_B,
#         "edge_width": 1.25,
#         "edge_opacity": 0.75,
#         "layer_label_color": WHITE,
#         "layer_label_size": 0.5,
#         "neuron_label_color": WHITE,
#     }
# 
#     def construct(self):
#         self.camera.background_color = BLACK
#         self.add_neurons()
#         #self.edge_security()  # turn on for continual_animation
#         self.add_edges()  # turn off for continual_animation
#         #self.label_layers()
#         #self.label_neurons()
#         #self.pulse_animation()
#         #self.pulse_animation_2()
#         #self.wiggle_animation()
#         #self.continual_animation()
#         #self.activate_neurons()
#         #self.forward_pass_animation()
#         self.text()
#         self.pulse_animation()
# 
#     def text(self):
#         title = MathTex(r'Q^{new}(s_{t}, a_{t})',  r' \leftarrow', r'Q^{new}(s_{t}, a_{t})', r' +', r'\alpha', r'(', r'r_{t+1}', r' + ', r'\gamma', r' \dot ', r'max_{a} Q(s_{t+1}, a)', r' - ', r'Q(s_{t}, a_{t})').scale(0.5)
#         title.shift(2*UP)
#         title.set_color(YELLOW_B)
# 
#         old_value = BraceLabel(obj=title[2], text="old value", brace_direction=DOWN)
#         old_value.height(0.25)
#         old_value.width(0.50)
# 
#         learning_rate = BraceLabel(obj=title[4], text="Learning rate", brace_direction=DOWN)
#         learning_rate.height(np.array())
#         learning_rate.width(0.50)
# 
#         reward = BraceLabel(obj=title[6], text="reward", brace_direction=DOWN)
#         reward.height(0.25)
#         reward.width(0.50)
# 
#         gamma = BraceLabel(obj=title[8], text="discount factor", brace_direction=DOWN)
#         gamma.height(0.25)
#         gamma.width(0.50)
# 
#         q_val = BraceLabel(obj=title[10], text="estimate of optimal future value", brace_direction=DOWN)
#         q_val.height(0.25)
#         q_val.width(0.50)
# 
#         old_q_val = BraceLabel(obj=title[12], text="old value", brace_direction=DOWN)
#         old_q_val.height(0.25)
#         old_q_val.width(0.50)
# 
#         credits = Tex("with random weights")
#         credits.shift(2*DOWN)
#         credits.set_color(YELLOW_B)
#         self.play(Write(title))
#         self.play(Write(old_value))
#         self.play(Write(learning_rate))
#         self.play(Write(reward))
#         self.play(Write(q_val))
#         self.play(Write(old_q_val))
# 
#     def add_neurons(self):
#         layers = VGroup(*[self.get_layer(size) for size in NeuralNetwork.arguments["layer_sizes"]])
#         layers.arrange(RIGHT, buff=NeuralNetwork.arguments["layer_buff"])
#         layers.scale(NeuralNetwork.arguments["network_size"])
#         # self.layers is layers, but we can use it throughout every method in our class
#         # without having to redefine layers each time
#         self.layers = layers
#         layers.shift(NeuralNetwork.arguments["network_position"])
#         self.play(FadeInFromPoint(layers, ORIGIN), run_time=2)
# 
#     def get_layer(self, size):
#         layer = VGroup()
#         n_neurons = size
#         neurons = VGroup(*[
#             Circle(
#                 radius=NeuralNetwork.arguments["neuron_radius"],
#                 stroke_color=NeuralNetwork.arguments["neuron_color"],
#                 stroke_width=NeuralNetwork.arguments["neuron_width"],
#                 fill_color=NeuralNetwork.arguments["neuron_fill_color"],
#                 fill_opacity=NeuralNetwork.arguments["neuron_fill_opacity"],
#             )
#             for i in range(n_neurons)
#         ])
#         neurons.arrange(DOWN, buff=NeuralNetwork.arguments["neuron_buff"])
#         layer.neurons = neurons
#         layer.add(neurons)
#         return layer
# 
#     def edge_security(self):
#         self.edge_groups = VGroup()
#         for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
#             edge_group = VGroup()
#             for n1, n2 in it.product(l1.neurons, l2.neurons):
#                 edge = self.get_edge(n1, n2)
#                 edge_group.add(edge)
#             self.edge_groups.add(edge_group)
# 
#     def add_edges(self):
#         self.edge_groups = VGroup()
#         for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
#             edge_group = VGroup()
#             for n1, n2 in it.product(l1.neurons, l2.neurons):
#                 edge = self.get_edge(n1, n2)
#                 edge_group.add(edge)
#             self.play(Write(edge_group), run_time=0.5)
#             self.edge_groups.add(edge_group)
# 
#     def get_edge(self, neuron1, neuron2):
#         return Line(
#             neuron1.get_center(),
#             neuron2.get_center(),
#             buff=1.25*NeuralNetwork.arguments["neuron_radius"],
#             stroke_color=NeuralNetwork.arguments["edge_color"],
#             stroke_width=NeuralNetwork.arguments["edge_width"],
#             stroke_opacity=NeuralNetwork.arguments["edge_opacity"]
#         )
# 
#     def label_layers(self):
#         input_layer = VGroup(*self.layers[0])
#         input_label = Tex("Input Layer")
#         input_label.set_color(NeuralNetwork.arguments["layer_label_color"])
#         input_label.scale(NeuralNetwork.arguments["layer_label_size"])
#         input_label.next_to(input_layer, UP, SMALL_BUFF)
#         self.play(Write(input_label))
# 
#         hidden_layer = VGroup(*self.layers[1:3])
#         hidden_label = Tex("Hidden Layers")
#         hidden_label.set_color(NeuralNetwork.arguments["layer_label_color"])
#         hidden_label.scale(NeuralNetwork.arguments["layer_label_size"])
#         hidden_label.next_to(hidden_layer, UP, SMALL_BUFF)
#         self.play(Write(hidden_label))
# 
#         output_layer = VGroup(*self.layers[-1])
#         output_label = Tex("Output Layer")
#         output_label.set_color(NeuralNetwork.arguments["layer_label_color"])
#         output_label.scale(NeuralNetwork.arguments["layer_label_size"])
#         output_label.next_to(output_layer, UP, SMALL_BUFF)
#         self.play(Write(output_label))
# 
#     def label_neurons(self):
#         input_labels = VGroup()
#         for n, neuron in enumerate(self.layers[0].neurons):
#             label = MathTex(f"x_{n + 1}")
#             label.set_height(0.3 * neuron.get_height())
#             label.set_color(NeuralNetwork.arguments["neuron_label_color"])
#             label.move_to(neuron)
#             input_labels.add(label)
#         self.play(Write(input_labels))
# 
#         weight_labels = VGroup()
#         for n, neuron in enumerate(self.layers[2].neurons):
#             label = MathTex(f"w_{n + 1}")
#             label.set_height(0.3 * neuron.get_height())
#             label.set_color(NeuralNetwork.arguments["neuron_label_color"])
#             label.move_to(neuron)
#             weight_labels.add(label)
#         self.play(Write(weight_labels))
# 
#         output_labels = VGroup()
#         for n, neuron in enumerate(self.layers[-1].neurons):
#             label = MathTex(r"\hat{y}_" + "{" + f"{n + 1}" + "}")
#             label.set_height(0.4 * neuron.get_height())
#             label.set_color(NeuralNetwork.arguments["neuron_label_color"])
#             label.move_to(neuron)
#             output_labels.add(label)
#         self.play(Write(output_labels))
# 
#     def pulse_animation(self):
#         edge_group = self.edge_groups.copy()
#         edge_group.set_stroke(MAROON_E, 2.5)  # color, width
#         for i in range(1):
#             self.play(LaggedStartMap(
#                 ShowCreationThenDestruction, edge_group))
#         self.wait()
# 
#     def pulse_animation_2(self):
#         edge_group = VGroup(*it.chain(*self.edge_groups))
#         edge_group = edge_group.copy()
#         edge_group.set_stroke(YELLOW, 4)  # color, width
#         for i in range(3):
#             self.play(LaggedStartMap(
#                 ShowCreationThenDestruction, edge_group,
#                 run_time=1.5))
#             self.wait()
# 
#     def wiggle_animation(self):
#         edges = VGroup(*it.chain(*self.edge_groups))
#         self.play(LaggedStartMap(
#             ApplyFunction, edges,
#             lambda mob: (lambda m: m.rotate_in_place(np.pi/12).set_color(YELLOW), mob),
#             rate_func=wiggle))
# 
#     def continual_animation(self):
#         args = {
#             "colors": [BLUE, BLUE, RED, RED],
#             "n_cycles": 5,
#             "max_width": 3,
#             "exp_width": 7
#         }
#         self.internal_time = 0
#         self.move_to_targets = []
#         edges = VGroup(*it.chain(*self.edge_groups))
#         for edge in edges:
#             edge.colors = [random.choice(args["colors"]) for i in range(args["n_cycles"])]
#             msw = args["max_width"]
#             edge.widths = [msw * random.random()**args["exp_width"] for i in range(args["n_cycles"])]
#             edge.cycle_time = 1 + random.random()
# 
#             edge.generate_target()
#             edge.target.set_stroke(edge.colors[0], edge.widths[0])
#             edge.become(edge.target)
#             self.move_to_targets.append(edge)
# 
#         self.edges = edges
#         animation = self.edges.add_updater(lambda m, dt: self.update_edges(dt))
#         self.play(ShowCreation(animation))
#         self.wait(5)
# 
#     def update_edges(self, dt):
#         self.internal_time += dt
#         if self.internal_time < 1:
#             alpha = smooth(self.internal_time)
#             for i in self.move_to_targets:
#                 i.update(alpha)
#             return
#         for edge in self.edges:
#             t = (self.internal_time-1)/edge.cycle_time
#             alpha = ((self.internal_time-1)%edge.cycle_time)/edge.cycle_time
#             low_n = int(t)%len(edge.colors)
#             high_n = int(t+1)%len(edge.colors)
#             color = interpolate_color(edge.colors[low_n], edge.colors[high_n], alpha)
#             width = interpolate(edge.widths[low_n], edge.widths[high_n], alpha)
#             edge.set_stroke(color, width)
# 
#     def forward_pass_animation(self):
#         edge_group = self.edge_groups.copy()
#         edge_group.set_stroke(red, 4)  # c1: red
# 
#         for i in range(len(self.layers)-1):
#             self.layers[i].neurons.set_fill(color=NeuralNetwork.arguments["neuron_color"], opacity=1)
#             self.play(FadeIn(self.layers[i].neurons))
#             self.play(LaggedStartMap(ShowCreationThenDestruction, edge_group[i]))
# 
#         self.layers[-1].neurons.set_fill(color=NeuralNetwork.arguments["neuron_color"], opacity=1)
#         self.play(FadeIn(self.layers[-1].neurons))

from google.colab import drive
drive.mount('/content/drive')

