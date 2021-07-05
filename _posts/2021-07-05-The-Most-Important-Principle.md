---
layout: post
title: The Most Important Principle Of Data/Computer Science
classes: wide
excerpt: "How to avoid the traps?"
image: /images/gigo.jpg
---

Reading some fitness guide, I stumbled upon, to my surprise, the most important principle of computer science: *GIGO*, or *Garbage-In-Garbage-Out*. 
The author implied that to have good training results (output), you must make sure that your training methods are good too (input) and vice versa: 
poor methods yield poor results. Obvious, right? Yet, it is too tempting to just start mindlessly working out (Nike's infamous "Just do it") 
instead of developing a sound strategy in the first place.

Fitness world aside, I then tried to remember any computer science, programming or data science lecture, book or tutorial that I had come across, and, to my astonishment, 
this emphatic and beautiful principle was not there to be found. Too obvious! And, we are way too smart, we people working with code/AI/<another-buzzword>, 
to be reminded of such, perhaps patronizing, "words of wisdom".

Well, at a risk of merely entertaining you, here is my attempt to stand up for GIGO and let it be a beacon, saving your data ships from crashing into coastal rocks.
As a machine learning engineer, one may be tempted to quickly label a business problem as classification, clustering, regression, computer vision, or other species,
before actually understanding the *real* problem itself. If you work as a consultant, I believe, your responsibility is to make the most of your data skills to increase the company profits. 
There are two ways to achieve this goal: increase the revenue and/or decrease the costs. Period. It is way too easy to focus on the former and skip the latter. 
For example, one may develop a sophisticated model that requires expensive infrastructure, increasing, perhaps, the turnover, but also, the costs. Meanwhile, a single-CPU solution would suffice. 
Therefore, even a seemingly meaningful dataset or approach may become a garbage, if it misses the business goal. 
Speaking with a project manager / CTO about the enterprise goals, business model and needs, and keeping this big picture would be the first GIGO beacon.

Then, once the objectives (ideally, including the budget) are well-defined, we may proceed to the next step: defining the data science project itself. That is to say, translating the business needs into the data science lingo (here comes classification, regression, sampling, cleaning, data ingestion). Some helpful steps and questions include:

- Finding bugs in data
- Asking "what does X column really mean"?
- How would you teach the ML task to a kid?
- Is the database structure OK or should we rebuild it?
- What will change in the future? How might the project scale up? What's the expected size of the database?

I might continue this post by laying out the end-to-end generic checklist for most data science projects, but that is not the point. 
Instead, I just wanted to suggest that it is primordial to always keep the GIGO beacon on, especially when we get too enthusiastic 
about optimizing our favourite model.

Here's an anecdote to sum it all up: my high school maths teacher, in case someone slightly misinterpreted the homework problem, but applied the correct method, 
would still give the worst grade, because, as she used to put it: "You solved your *OWN* problem. That's not *MY* problem".

Let the GIGO beacon guide us all.