---
title: Conv1d 与 Conv2d 区别
url: https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d
fetch_source: agent_reach:jina
fetched_at: '2026-06-27T17:54:26+00:00'
polished_at: '2026-06-27T18:51:39+00:00'
math_repaired_at: '2026-06-27T20:24:23+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# Conv1d 与 Conv2d 区别

[Skip to main content](https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d#content) [](https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d#)[](https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d#)

#### Stack Exchange Network
Stack Exchange network consists of 184 Q&A communities including [Stack Overflow](https://stackoverflow.com/), the largest, most trusted online community for developers to learn, share their knowledge, and build their careers.

[Visit Stack Exchange](https://stackexchange.com/) Loading…
1.   [](https://stats.stackexchange.com/help "Help Center and other resources")
    *   [Tour Start here for a quick overview of the site](https://stats.stackexchange.com/tour)     *   [Help Center Detailed answers to any questions you might have](https://stats.stackexchange.com/help)     *   [Meta Discuss the workings and policies of this site](https://stats.meta.stackexchange.com/)     *   [About Us Learn more about Stack Overflow the company, and our products](https://stackoverflow.co/)
2.   [](https://stackexchange.com/ "A list of all 184 Stack Exchange sites")

3.

### [current community](https://stats.stackexchange.com/)
    *   [Cross Validated](https://stats.stackexchange.com/) [help](https://stats.stackexchange.com/help)[chat](https://chat.stackexchange.com/?tab=site&host=stats.stackexchange.com)     *    [Cross Validated Meta](https://stats.meta.stackexchange.com/)
### your communities

[Sign up](https://stats.stackexchange.com/users/signup?ssrc=site_switcher&returnurl=https%3a%2f%2fstats.stackexchange.com%2fquestions%2f295397%2fwhat-is-the-difference-between-conv1d-and-conv2d) or [log in](https://stats.stackexchange.com/users/login?ssrc=site_switcher&returnurl=https%3a%2f%2fstats.stackexchange.com%2fquestions%2f295397%2fwhat-is-the-difference-between-conv1d-and-conv2d) to customize your list.

### [more stack exchange communities](https://stackexchange.com/sites)
[company blog](https://stackoverflow.blog/)
5.   [Log in](https://stats.stackexchange.com/users/login?ssrc=head&returnurl=https%3a%2f%2fstats.stackexchange.com%2fquestions%2f295397%2fwhat-is-the-difference-between-conv1d-and-conv2d)
6.   [Sign up](https://stats.stackexchange.com/users/signup?ssrc=head&returnurl=https%3a%2f%2fstats.stackexchange.com%2fquestions%2f295397%2fwhat-is-the-difference-between-conv1d-and-conv2d)

[![Image 3: Cross Validated](https://stats.stackexchange.com/Content/Sites/stats/Img/logo.svg?v=60d6be2c448d)](https://stats.stackexchange.com/)

<p class="kb-image-caption">图例</p>

    2.   [Questions](https://stats.stackexchange.com/questions)     3.   [Unanswered](https://stats.stackexchange.com/unanswered)     4.   [AI Assist](https://stackoverflow.com/ai-assist)     5.   [Tags](https://stats.stackexchange.com/tags)     7.   [Chat](https://chat.stackexchange.com/)     8.   [Users](https://stats.stackexchange.com/users)
2.   Stack Internal Stack Overflow for Teams is now called **Stack Internal**. Bring the best of human thought and AI automation together at your work.

[Try for free](https://stackoverflowteams.com/teams/create/free/?utm_medium=referral&utm_source=stats-community&utm_campaign=side-bar&utm_content=explore-teams)[Learn more](https://stackoverflow.co/internal/?utm_medium=referral&utm_source=stats-community&utm_campaign=side-bar&utm_content=explore-teams)
3.   [Stack Internal](javascript:void(0))
4.   Bring the best of human thought and AI automation together at your work. [Learn more](https://stackoverflow.co/internal/?utm_medium=referral&utm_source=stats-community&utm_campaign=side-bar&utm_content=explore-teams-compact)

**Stack Internal**

Knowledge at work Bring the best of human thought and AI automation together at your work.

[Explore Stack Internal](https://stackoverflow.co/internal/?utm_medium=referral&utm_source=stats-community&utm_campaign=side-bar&utm_content=explore-teams-compact-popover)

# [What is the difference between Conv1D and Conv2D?](https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d)
[Ask Question](https://stats.stackexchange.com/questions/ask) Asked 8 years, 11 months ago Modified[1 year, 11 months ago](https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d?lastactivity "2024-06-30 12:44:50Z") Viewed 87k times  43 [](https://stats.stackexchange.com/posts/295397/timeline "Show activity on this post.") I was going through the keras convolution [docs](https://keras.io/layers/convolutional/) and I have found two types of convultuion Conv1D and Conv2D. I did some web search and this is what I understands about Conv1D and Conv2D; Conv1D is used for sequences and Conv2D uses for images.

I always thought convolution nerual networks were used only for images and visualized CNN this way

[![Image 4: enter image description here](https://i.sstatic.net/xdXTn.gif)](https://i.sstatic.net/xdXTn.gif)

<p class="kb-image-caption">图例</p>

A image is considered as a large matrix and then a filter will slide over this matrix and compute the dot product. This I believe what keras mentions as a Conv2D. If Conv2D works this way then what is the mechanism of Conv1D and how we can imagine its mechanism?
*   [machine-learning](https://stats.stackexchange.com/questions/tagged/machine-learning "show questions tagged 'machine-learning'")
*   [neural-networks](https://stats.stackexchange.com/questions/tagged/neural-networks "show questions tagged 'neural-networks'")
*   [convolutional-neural-network](https://stats.stackexchange.com/questions/tagged/convolutional-neural-network "show questions tagged 'convolutional-neural-network'")
*   [keras](https://stats.stackexchange.com/questions/tagged/keras "show questions tagged 'keras'")
*   2  Take a look at this [answer](https://stackoverflow.com/questions/42883547/what-do-you-mean-by-1d-2d-and-3d-convolutions-in-cnn). Hope this helps.learner101  –[learner101](https://stats.stackexchange.com/users/182593/learner101 "1 reputation") 2017-10-28 00:03:35 +00:00 Commented Oct 28, 2017 at 0:03

[Add a comment](https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d# "Use comments to ask for more information or suggest improvements. Avoid answering questions in comments.")|[](https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d# "Expand to show all comments on this post") [](https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d)

## 5 Answers 5

 Sorted by:  [Reset to default](https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d?answertab=scoredesc#tab-top) [](https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d)  18 [](https://stats.stackexchange.com/posts/454115/timeline "Show activity on this post.") I'd like to explain the difference visually and in detail and in a very easy approach.

Let's first check [the Conv2D in TensorFlow](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d).

[![Image 7: enter image description here](https://i.sstatic.net/uwHol.gif)](https://i.sstatic.net/uwHol.gif)

<p class="kb-image-caption">图例</p>

c1 = [[0, 0, 1, 0, 2], [1, 0, 2, 0, 1], [1, 0, 2, 2, 0], [2, 0, 0, 2, 0], [2, 1, 2, 2, 0]] c2 = [[2, 1, 2, 1, 1], [2, 1, 2, 0, 1], [0, 2, 1, 0, 1], [1, 2, 2, 2, 2], [0, 1, 2, 0, 1]] c3 = [[2, 1, 1, 2, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 0], [1, 0, 2, 1, 0], [2, 2, 1, 1, 1]] data = tf.transpose(tf.constant(`[[c1, c2, c3]]`, dtype=tf.float32), (0, 2, 3, 1))

# we transfer [batch, in_channels, in_height, in_width] to
# [batch, in_height, in_width, in_channels]
# where batch = 1,
# in_channels = 3 (c1, c2, c3 or x[:, :, 0], x[:, :, 1], x[:, :, 2] in the gif)
# in_height and in_width are all 5 (the sizes of the blue matrices without padding)
f2c1 = [[0, 1, -1], [0, -1, 0], [0, -1, 1]] f2c2 = [[-1, 0, 0], [1, -1, 0], [1, -1, 0]] f2c3 = [[-1, 1, -1], [0, -1, -1], [1, 0, 0]] filters = tf.transpose(tf.constant(`[[f2c1, f2c2, f2c3]]`, dtype=tf.float32), (2, 3, 1, 0))

# transfer the [out_channels, in_channels, filter_height, filter_width] to
# [filter_height, filter_width, in_channels, out_channels]
# out_channels is 1 (in the gif it is 2 since here we only use one filter W1),
# in_channels is 3 because data has three channels(c1, c2, c3),
# filter_height and filter_width are all 3 (the sizes of the filter W1)
# f2c1, f2c2, f2c3 are the w1[:, :, 0], w1[:, :, 1] and w1[:, :, 2] in the gif
output = tf.squeeze(tf.nn.conv2d(data, filters, strides=2, padding=[[0, 0], [1, 1], [1, 1], [0, 0]]))

# this is just the o[:,:,1] in the gif
# <tf.Tensor: id=93, shape=(3, 3), dtype=float32, numpy=
# array([[-8., -8., -3.],
#        [-3.,  1.,  0.],
#        [-3., -8., -5.]], dtype=float32)>
And the Conv1D is a special case of Conv2D as stated in this paragraph from [the TensorFlow doc of Conv1D](https://www.tensorflow.org/api_docs/python/tf/nn/conv1d).
> Internally, this op reshapes the input tensors and invokes tf.nn.conv2d. For example, if data_format does not start with "NC", a tensor of shape [batch, in_width, in_channels] is reshaped to [batch, 1, in_width, in_channels], and the filter is reshaped to [1, filter_width, in_channels, out_channels]. The result is then reshaped back to [batch, out_width, out_channels] (where out_width is a function of the stride and padding as in conv2d) and returned to the caller.

Let's see how we can transfer Conv1D to a Conv2D problem. Since Conv1D is usually used in NLP scenarios, we can illustrate that in the below NLP problem.

[![Image 8: enter image description here](https://i.sstatic.net/eWWP3.png)](https://i.sstatic.net/eWWP3.png)

<p class="kb-image-caption">图例</p>

cat = [0.7, 0.4, 0.5] sitting = [0.2, -0.1, 0.1] there = [-0.5, 0.4, 0.1] dog = [0.6, 0.3, 0.5] resting = [0.3, -0.1, 0.2] here = [-0.5, 0.4, 0.1] sentence = tf.constant(`[[cat, sitting, there, dog, resting, here]]`

# sentence[:,:,0] is equivalent to x[:,:,0] or c1 in the first example
# and the same for sentence[:,:,1] and sentence[:,:,2]
data = tf.reshape(sentence), (1, 1, 6, 3))

# we reshape [batch, in_width, in_channels] to
# [batch, 1, in_width, in_channels] according to the quote above
# each dimension in the embedding is a channel(three in_channels)
f3c1 = [0.6, 0.2]

# equivalent to f2c1 in the first code snippet or w1[:,:,0] in the gif
f3c2 = [0.4, -0.1]

# equivalent to f2c2 in the first code snippet or w1[:,:,1] in the gif
f3c3 = [0.5, 0.2]

# equivalent to f2c3 in the first code snippet or w1[:,:,2] in the gif
# filters = tf.constant(`[[f3c1, f3c2, f3c3]]`)
# [out_channels, in_channels, filter_width]: [1, 3, 2]
# here we also have only one filter and also three channels in it.
# Please compare these three with the three channels in W1 for the Conv2D in the gif
filter1D = tf.transpose(tf.constant(`[[f3c1, f3c2, f3c3]]`), (2, 1, 0))

# shape: [2, 3, 1] for the conv1d example
filters = tf.reshape(filter1D, (1, 2, 3, 1))  # this should be expand_dim actually

# transpose [out_channels, in_channels, filter_width] to
# [filter_width, in_channels, out_channels]]
# and then reshape the result to [1, filter_width, in_channels, out_channels]
# as we described in the text snippet from TensorFlow doc of conv1doutput
output = tf.squeeze(tf.nn.conv2d(data, filters, strides=(1, 1, 2, 1), padding="VALID"))

# the numbers for strides are for [batch, 1, in_width, in_channels] of the data input
# <tf.Tensor: id=119, shape=(3,), dtype=float32, numpy=array([0.9       , 0.09999999, 0.12      ], dtype=float32)>
Let's do that using Conv1D(also in TensorFlow):
output = tf.squeeze(tf.nn.conv1d(sentence, filter1D, stride=2, padding="VALID"))

# <tf.Tensor: id=135, shape=(3,), dtype=float32, numpy=array([0.9       , 0.09999999, 0.12      ], dtype=float32)>
# here stride defaults to be for the in_width
We can see that the 2D in Conv2D means each channel in the input and filter is 2-dimensional (as we see in the gif example) and 1D in Conv1D means each channel in the input and filter is 1 dimensional (as we see in the cat and dog NLP example).

When using Conv1d(), we have to keep in mind that we are most likely going to work with 2-dimensional inputs such as one-hot-encode DNA sequences or black and white pictures.

The only difference between the more conventional Conv2d() and Conv1d() is that latter uses a 1-dimensional kernel as shown in the picture below.

We can see that the kernel automatically spans to the height of the picture (just as in Conv2d() the depth of the kernel automatically spans the image’s channels) and therefore all we are left to give is the kernel size with respect to the span of the rows.

We just have to remember that if we are assuming a 2-dimensional input, our filters become our columns and our rows become the kernel size.
*     Picture was taken from this previous question: [stackoverflow.com/questions/48859378/…](https://stackoverflow.com/questions/48859378/how-to-give-the-1d-input-to-convolutional-neural-networkcnn-using-keras/52508449 "how to give the 1d input to convolutional neural networkcnn using keras")Erick Platero  –[Erick Platero](https://stats.stackexchange.com/users/246357/erick-platero "71 reputation") 2019-04-29 14:58:40 +00:00 Commented Apr 29, 2019 at 14:58

[Add a comment](https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d# "Use comments to ask for more information or suggest improvements. Avoid comments like “+1” or “thanks”.")|[](https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d# "Expand to show all comments on this post") [](https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d)  6 [](https://stats.stackexchange.com/posts/405689/timeline "Show activity on this post.") Convolution is a mathematical operation where you "summarize" a tensor or a matrix or a vector into a smaller one. If your input matrix is one dimensional then you summarize along that on dimensions, and if a tensor has n dimensions then you _could_ summarize along all n dimensions. Conv1D and Conv2D summarize (convolve) along one or two dimensions.

For instance, you could convolve a vector into a shorter vector as followss. Get a "long" vector A with n elements and convolve it using the a weight vector W with m elements into a "short" (summary) vector B with n-m+1 elements:
b i=∑j=m−1 0 a i+j∗w j $$ b_{i} = \sum_{j = m - 1}^{0} a_{i + j} * w_{j} $$ where i=[1,n−m+1]$i = \left[ 1 , n - m + 1 \right]$ So, if you have vector of length n, and your weight matrix is also length n w i=1/n$w_{i} = 1 / n$, then the convolution will produce a scalar or a vector of length 1 equal to the average value of all values in the input matrix. It's a sort of degenerate convolution if you wish. If the same weight matrix is one shorter than the input matrix, then you get a moving average in the output of length 2 etc.
⎡⎣⎢a:w:w:a 1 1/2 a 2 1/2 1/2 a 3 1/2⎤⎦⎥=[b:a 1+a 2 2 a 2+a 3 2] $$ \left[ a : & a_{1} & a_{2} & a_{3} \\ w : & 1 / 2 & 1 / 2 & \\ w : & & 1 / 2 & 1 / 2 \right] = \left[ b : & \frac{a_{1} + a_{2}}{2} & \frac{a_{2} + a_{3}}{2} \right]
$$ You could do the same to 3 dimensional tensor (matrix) the same way:
In 2D CNN, kernel moves in 2 directions. Input and output data of 2D CNN is 3 dimensional. Mostly used on Image data.

In 3D CNN, kernel moves in 3 directions. Input and output data of 3D CNN is 4 dimensional. Mostly used on 3D Image data (MRI, CT Scans).

## Your Answer

Thanks for contributing an answer to Cross Validated!
*   Please be sure to _answer the question_. Provide details and share your research!

But _avoid_ …
*   Asking for help, clarification, or responding to other answers.
*   Making statements based on opinion; back them up with references or personal experience.

Use MathJax to format equations. [MathJax reference](https://stats.meta.stackexchange.com/questions/1604/instructions-on-how-to-use-latex-on-crossvalidated).

To learn more, see our [tips on writing great answers](https://stats.stackexchange.com/help/how-to-answer).

Draft saved Draft discarded

### Sign up or [log in](https://stats.stackexchange.com/users/login?ssrc=question_page&returnurl=https%3a%2f%2fstats.stackexchange.com%2fquestions%2f295397%2fwhat-is-the-difference-between-conv1d-and-conv2d%23new-answer)
 Sign up using Google  Sign up using Email and Password Submit

### Post as a guest

Name Email Required, but never shown  Post Your Answer  Discard By clicking “Post Your Answer”, you agree to our [terms of service](https://stackoverflow.com/legal/terms-of-service/public) and acknowledge you have read our [privacy policy](https://stackoverflow.com/legal/privacy-policy).

Start asking to get answers Find the answer to your question by asking.

[Ask question](https://stats.stackexchange.com/questions/ask) Explore related questions
See similar questions with these tags.
*    Featured on Meta
*     [Partnering with Communities to Modernize Policies & Norms](https://meta.stackexchange.com/questions/418826/partnering-with-communities-to-modernize-policies-norms)
### Linked

[41](https://stats.stackexchange.com/questions/296679/what-does-kernel-size-mean "Question score (upvotes - downvotes)")[What does kernel size mean?](https://stats.stackexchange.com/questions/296679/what-does-kernel-size-mean?noredirect=1)

### Related

[0](https://stats.stackexchange.com/questions/300082/how-many-times-filter-multiplied-by-rgb-images-which-has-three-dimension "Question score (upvotes - downvotes)")[How many times filter multiplied by RGB images which has three dimension?](https://stats.stackexchange.com/questions/300082/how-many-times-filter-multiplied-by-rgb-images-which-has-three-dimension) [1](https://stats.stackexchange.com/questions/310288/can-i-convolve-over-spatially-remote-pixels-in-1dconv-in-keras "Question score (upvotes - downvotes)")[Can I convolve over spatially remote pixels in 1DConv in Keras](https://stats.stackexchange.com/questions/310288/can-i-convolve-over-spatially-remote-pixels-in-1dconv-in-keras) [23](https://stats.stackexchange.com/questions/335321/in-a-convolutional-neural-network-cnn-when-convolving-the-image-is-the-opera "Question score (upvotes - downvotes)")[In a convolutional neural network (CNN), when convolving the image, is the operation used the dot product or the sum of element-wise multiplication?](https://stats.stackexchange.com/questions/335321/in-a-convolutional-neural-network-cnn-when-convolving-the-image-is-the-opera) [1](https://stats.stackexchange.com/questions/407927/1-d-convolution-neural-network-in-keras "Question score (upvotes - downvotes)")[1-D convolution neural network in Keras](https://stats.stackexchange.com/questions/407927/1-d-convolution-neural-network-in-keras) [2](https://stats.stackexchange.com/questions/551194/what-is-the-best-way-to-feed-imu-data-to-cnn "Question score (upvotes - downvotes)")[What is the best way to feed IMU data to CNN?](https://stats.stackexchange.com/questions/551194/what-is-the-best-way-to-feed-imu-data-to-cnn) [1](https://stats.stackexchange.com/questions/571138/implementation-of-a-convolution-layer-in-a-cnn "Question score (upvotes - downvotes)")[Implementation of a convolution layer in a cnn](https://stats.stackexchange.com/questions/571138/implementation-of-a-convolution-layer-in-a-cnn)

#### [Hot Network Questions](https://stackexchange.com/questions?tab=hot)
*    [Why 2N7002 on I2C bus?](https://electronics.stackexchange.com/questions/770211/why-2n7002-on-i2c-bus)
*    [Beatrice's experiment with mirrors in canto 2 of Dante's "Paradiso"](https://literature.stackexchange.com/questions/32135/beatrices-experiment-with-mirrors-in-canto-2-of-dantes-paradiso)
*    [How can I stop relying on AI when writing code?](https://stackoverflow.com/questions/79964882/how-can-i-stop-relying-on-ai-when-writing-code)
*    [How to prevent `Dt` from applying chain rule to some inner functions?](https://mathematica.stackexchange.com/questions/319645/how-to-prevent-dt-from-applying-chain-rule-to-some-inner-functions)
*    [How/could one figure out any method of variolation for an unknown water born virus without killing too many in the process?](https://worldbuilding.stackexchange.com/questions/273871/how-could-one-figure-out-any-method-of-variolation-for-an-unknown-water-born-vir)
*    [Why are interpretations like Copenhagen and Many Worlds in quantum mechanics taken seriously as science?](https://philosophy.stackexchange.com/questions/139315/why-are-interpretations-like-copenhagen-and-many-worlds-in-quantum-mechanics-tak)
*    [Practical maximum clique search on 300–900 vertex graphs with only a few seconds available](https://mathoverflow.net/questions/512587/practical-maximum-clique-search-on-300-900-vertex-graphs-with-only-a-few-seconds)
*    ["Exact" Voronoi cells on sphere](https://mathematica.stackexchange.com/questions/319649/exact-voronoi-cells-on-sphere)
*    [Fractal Geometry: In this Fourier series substitution, how do I finish the calculation?](https://math.stackexchange.com/questions/5141932/fractal-geometry-in-this-fourier-series-substitution-how-do-i-finish-the-calcu)
*    [How do I fix this aluminium door / who should I call?](https://diy.stackexchange.com/questions/331088/how-do-i-fix-this-aluminium-door-who-should-i-call)
*    [What precisely do we mean when we say that an atom has a certain electron configuration?](https://physics.stackexchange.com/questions/873741/what-precisely-do-we-mean-when-we-say-that-an-atom-has-a-certain-electron-config)
*    [Can I reapply for an ESTA after overstaying by 1 day due to grieving?](https://travel.stackexchange.com/questions/203953/can-i-reapply-for-an-esta-after-overstaying-by-1-day-due-to-grieving)
*    [Context: How to add small font verbatim text to image caption?](https://tex.stackexchange.com/questions/764211/context-how-to-add-small-font-verbatim-text-to-image-caption)
*    [Using two different fonts in math mode with LuaLaTeX](https://tex.stackexchange.com/questions/764217/using-two-different-fonts-in-math-mode-with-lualatex)
*    [Understanding an RF circuit on the ESP32 S3 mini](https://electronics.stackexchange.com/questions/770227/understanding-an-rf-circuit-on-the-esp32-s3-mini)
*    [Should the bottom of a chromoly steerer tube be closed from water if it's open initially, to prevent rust?](https://bicycles.stackexchange.com/questions/100433/should-the-bottom-of-a-chromoly-steerer-tube-be-closed-from-water-if-its-open-i)
*    [What happens when a choice doesn't matter (until it does)?](https://boardgames.stackexchange.com/questions/64544/what-happens-when-a-choice-doesnt-matter-until-it-does)
*    [Are some people zombies?](https://philosophy.stackexchange.com/questions/139352/are-some-people-zombies)
*    [Games night (Part 1)](https://puzzling.stackexchange.com/questions/138670/games-night-part-1)
*    [How can I create properly connected Persian/Arabic text using Geometry Nodes?](https://blender.stackexchange.com/questions/347337/how-can-i-create-properly-connected-persian-arabic-text-using-geometry-nodes)
*    [Can an exalted trigger a scene long charm in anticipation of combat?](https://rpg.stackexchange.com/questions/219539/can-an-exalted-trigger-a-scene-long-charm-in-anticipation-of-combat)
*    [Cobalt blue... and "fish blue"](https://puzzling.stackexchange.com/questions/138667/cobalt-blue-and-fish-blue)
*    [Why did Littlefinger do this?](https://movies.stackexchange.com/questions/132069/why-did-littlefinger-do-this)
*    [How is learning metaphysics useful for spiritual progress?](https://hinduism.stackexchange.com/questions/70137/how-is-learning-metaphysics-useful-for-spiritual-progress)

[more hot questions](https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d#) [Question feed](https://stats.stackexchange.com/feeds/question/295397 "Feed of this question and its answers")

# Subscribe to RSS

 Question feed To subscribe to this RSS feed, copy and paste this URL into your RSS reader.

[](https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d#) lang-py

##### [Cross Validated](https://stats.stackexchange.com/)
*   [Tour](https://stats.stackexchange.com/tour)
*   [Help](https://stats.stackexchange.com/help)
*   [Chat](https://chat.stackexchange.com/?tab=site&host=stats.stackexchange.com)
*   [Contact](https://stats.stackexchange.com/contact)
*   [Feedback](https://stats.meta.stackexchange.com/)
##### [Company](https://stackoverflow.co/)
*   [Stack Overflow](https://stackoverflow.com/)
*   [Stack Internal](https://stackoverflow.co/internal/)
*   [Stack Data Licensing](https://stackoverflow.co/data-licensing/)
*   [Stack Ads](https://stackoverflow.co/advertising/)
*   [About](https://stackoverflow.co/)
*   [Press](https://stackoverflow.co/company/press/)
*   [Legal](https://stackoverflow.com/legal)
*   [Privacy Policy](https://stackoverflow.com/legal/privacy-policy)
*   [Terms of Service](https://stackoverflow.com/legal/terms-of-service/public)
*    Your Privacy Choices
*   [Cookie Policy](https://policies.stackoverflow.co/stack-overflow/cookie-policy)
##### [Stack Exchange Network](https://stackexchange.com/)
*   [Technology](https://stackexchange.com/sites#technology)
*   [Culture & recreation](https://stackexchange.com/sites#culturerecreation)
*   [Life & arts](https://stackexchange.com/sites#lifearts)
*   [Science](https://stackexchange.com/sites#science)
*   [Professional](https://stackexchange.com/sites#professional)
*   [Business](https://stackexchange.com/sites#business)
*   [API](https://api.stackexchange.com/)
*   [Data](https://data.stackexchange.com/)
*   [Blog](https://stackoverflow.blog/?blb=1)
*   [Facebook](https://www.facebook.com/officialstackoverflow/)
*   [Twitter](https://twitter.com/stackoverflow)
*   [LinkedIn](https://linkedin.com/company/stack-overflow)
*   [Instagram](https://www.instagram.com/thestackoverflow)

Site design / logo © 2026 Stack Exchange Inc; user contributions licensed under [CC BY-SA](https://stackoverflow.com/help/licensing). rev 2026.6.25.43791 By continuing to use this website, you agree Stack Exchange can store cookies on your device and disclose information in accordance with our [Cookie Policy](https://policies.stackoverflow.co/stack-overflow/cookie-policy/). By exiting this window, default cookies will be accepted. To reject cookies, select an option from below.
![Image 17: Stack Exchange Inc.](https://cdn.cookielaw.org/logos/static/ot_company_logo.png)

<p class="kb-image-caption">图例</p>

When you visit any of our websites, it may store or retrieve information on your browser, mostly in the form of cookies. This information might be about you, your preferences, or your device and is mostly used to make the site work as you expect it to. The information does not usually directly identify you, but it can give you a more personalized experience. Because we respect your right to privacy, you can choose not to allow some types of cookies. Click on the different category headings to find out more and manage your preferences. Please note, blocking some types of cookies may impact your experience of the site and the services we are able to offer.

[Cookie policy](https://policies.stackoverflow.co/stack-overflow/cookie-policy/) Accept all cookies

### Manage consent preferences
#### Strictly Necessary Cookies
Always Active These cookies are necessary for the website to function and cannot be switched off in our systems. They are usually only set in response to actions made by you which amount to a request for services, such as setting your privacy preferences, logging in or filling in forms. You can set your browser to block or alert you about these cookies, but some parts of the site will not then work. These cookies do not store any personally identifiable information.

#### Targeting Cookies
- [x] Targeting Cookies

These cookies are used to make advertising messages more relevant to you and may be set through our site by us or by our advertising partners. They may be used to build a profile of your interests and show you relevant advertising on our site or on other sites. They do not store directly personal information, but are based on uniquely identifying your browser and internet device.

#### Performance Cookies
- [x] Performance Cookies

These cookies allow us to count visits and traffic sources so we can measure and improve the performance of our site. They help us to know which pages are the most and least popular and see how visitors move around the site. All information these cookies collect is aggregated and therefore anonymous. If you do not allow these cookies we will not know when you have visited our site, and will not be able to monitor its performance.

#### Functional Cookies
- [x] Functional Cookies

These cookies enable the website to provide enhanced functionality and personalisation. They may be set by us or by third party providers whose services we have added to our pages. If you do not allow these cookies then some or all of these services may not function properly.

### Cookie List
Clear
*   - [x] checkbox label label

Apply Cancel Consent Leg.Interest
- [x] checkbox label label
- [x] checkbox label label
- [x] checkbox label label

![Image 19: .](https://ams-pageview-public.s3.amazonaws.com/1x1-pixel.png?id=b1ffe3826ebc)

<p class="kb-image-caption">图例</p>
