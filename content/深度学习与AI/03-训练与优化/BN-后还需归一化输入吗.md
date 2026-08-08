---
title: BN 后还需归一化输入吗
url: https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used
fetch_source: agent_reach:jina
fetched_at: '2026-06-27T17:57:17+00:00'
polished_at: '2026-06-27T18:51:39+00:00'
math_repaired_at: '2026-06-27T20:24:23+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# BN 后还需归一化输入吗

[Skip to main content](https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used#content) [](https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used#)[](https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used#)

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

[Sign up](https://stats.stackexchange.com/users/signup?ssrc=site_switcher&returnurl=https%3a%2f%2fstats.stackexchange.com%2fquestions%2f249378%2fis-scaling-data-0-1-necessary-when-batch-normalization-is-used) or [log in](https://stats.stackexchange.com/users/login?ssrc=site_switcher&returnurl=https%3a%2f%2fstats.stackexchange.com%2fquestions%2f249378%2fis-scaling-data-0-1-necessary-when-batch-normalization-is-used) to customize your list.

### [more stack exchange communities](https://stackexchange.com/sites)
[company blog](https://stackoverflow.blog/)
5.   [Log in](https://stats.stackexchange.com/users/login?ssrc=head&returnurl=https%3a%2f%2fstats.stackexchange.com%2fquestions%2f249378%2fis-scaling-data-0-1-necessary-when-batch-normalization-is-used)
6.   [Sign up](https://stats.stackexchange.com/users/signup?ssrc=head&returnurl=https%3a%2f%2fstats.stackexchange.com%2fquestions%2f249378%2fis-scaling-data-0-1-necessary-when-batch-normalization-is-used)

[![Image 5: Cross Validated](https://stats.stackexchange.com/Content/Sites/stats/Img/logo.svg?v=60d6be2c448d)](https://stats.stackexchange.com/)

<p class="kb-image-caption">图例</p>

    2.   [Questions](https://stats.stackexchange.com/questions)     3.   [Unanswered](https://stats.stackexchange.com/unanswered)     4.   [AI Assist](https://stackoverflow.com/ai-assist)     5.   [Tags](https://stats.stackexchange.com/tags)     7.   [Chat](https://chat.stackexchange.com/)     8.   [Users](https://stats.stackexchange.com/users)
2.   Stack Internal Stack Overflow for Teams is now called **Stack Internal**. Bring the best of human thought and AI automation together at your work.

[Try for free](https://stackoverflowteams.com/teams/create/free/?utm_medium=referral&utm_source=stats-community&utm_campaign=side-bar&utm_content=explore-teams)[Learn more](https://stackoverflow.co/internal/?utm_medium=referral&utm_source=stats-community&utm_campaign=side-bar&utm_content=explore-teams)
3.   [Stack Internal](javascript:void(0))
4.   Bring the best of human thought and AI automation together at your work. [Learn more](https://stackoverflow.co/internal/?utm_medium=referral&utm_source=stats-community&utm_campaign=side-bar&utm_content=explore-teams-compact)

**Stack Internal**

Knowledge at work Bring the best of human thought and AI automation together at your work.

[Explore Stack Internal](https://stackoverflow.co/internal/?utm_medium=referral&utm_source=stats-community&utm_campaign=side-bar&utm_content=explore-teams-compact-popover)

# [is scaling data [0,1] necessary when batch normalization is used?](https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used)
[Ask Question](https://stats.stackexchange.com/questions/ask) Asked 9 years, 6 months ago Modified[5 years, 8 months ago](https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used?lastactivity "2020-10-16 14:26:08Z") Viewed 15k times Report this ad  22 [](https://stats.stackexchange.com/posts/249378/timeline "Show activity on this post.") Although a Relu activation function can deal with real value number but I have tried scaling the dataset in the range [0,1] (min-max scaling) is more effective before feed it to the neural network. on the other hand, the batch normalization (BN) is also normalizing data before passed to the non-linearity layer (activation function). I was wondering if the min-max scaling is still needed when BN is applied. can we perform min-max scaling and BN together?. It would be nice if someone guides me to the better understanding
*   [regression](https://stats.stackexchange.com/questions/tagged/regression "show questions tagged 'regression'")
*   [neural-networks](https://stats.stackexchange.com/questions/tagged/neural-networks "show questions tagged 'neural-networks'")
*   [normalization](https://stats.stackexchange.com/questions/tagged/normalization "show questions tagged 'normalization'")
*   [batch-normalization](https://stats.stackexchange.com/questions/tagged/batch-normalization "show questions tagged 'batch-normalization'")
*     I think you can let go of batch normalization. It is not needed any more.Souradeep Nanda  –[Souradeep Nanda](https://stats.stackexchange.com/users/123840/souradeep-nanda "595 reputation") 2016-12-05 05:39:11 +00:00 Commented Dec 5, 2016 at 5:39

[Add a comment](https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used# "Use comments to ask for more information or suggest improvements. Avoid answering questions in comments.")|[](https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used# "Expand to show all comments on this post") [](https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used)

## 2 Answers 2

 Sorted by:  [Reset to default](https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used?answertab=scoredesc#tab-top) [](https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used)  11 [](https://stats.stackexchange.com/posts/328988/timeline "Show activity on this post.") As mentioned, it's best to use [-1, 1] min-max scaling or zero-mean, unit-variance standardization. Scaling your data into [0, 1] will result in slow learning.

To answer your question: **Yes**, you should still standardize your inputs to a network that uses Batch Normalization. This will ensure that inputs to the first layer have zero mean and come from the same distribution, while Batch Normalization on subsequent layers will ensure that inputs to those layers have zero mean in expectation and that their distributions do not drift over time.

The reasons that we want zero mean and stable input distribution are discussed further in Section 4.3 of [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf).
*   5  As a hack, can I use batch norm in the first layer of my network? I am feeling lazy to preprocess the input :P Black Jack 21  –[Black Jack 21](https://stats.stackexchange.com/users/221304/black-jack-21 "103 reputation") 2020-03-15 07:04:10 +00:00 Commented Mar 15, 2020 at 7:04
*   2  Yes, see the "Alternate to Data Preparation" section in [machinelearningmastery.com/…](https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/). And if I remember correctly, Jeremy Howard also mentions this technique in the first edition of his Practical Deep Learning for Coders video series.Imran  –[Imran](https://stats.stackexchange.com/users/1104/imran "639 reputation") 2020-06-23 21:58:18 +00:00 Commented Jun 23, 2020 at 21:58

[Add a comment](https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used# "Use comments to ask for more information or suggest improvements. Avoid comments like “+1” or “thanks”.")|[](https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used# "Expand to show all comments on this post") [](https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used)  3 [](https://stats.stackexchange.com/posts/328978/timeline "Show activity on this post.") In this case scaling data would only influence the first layer of your network. Also if you are scaling your input it's better to scale it to [-1, 1], but it's best to scale it to 0 mean and 1 variance (since your weights are probably initialized to expect such distribution).

Not that it's going to make a huge difference anyway.

## Your Answer

Thanks for contributing an answer to Cross Validated!
*   Please be sure to _answer the question_. Provide details and share your research!

But _avoid_ …
*   Asking for help, clarification, or responding to other answers.
*   Making statements based on opinion; back them up with references or personal experience.

Use MathJax to format equations. [MathJax reference](https://stats.meta.stackexchange.com/questions/1604/instructions-on-how-to-use-latex-on-crossvalidated).

To learn more, see our [tips on writing great answers](https://stats.stackexchange.com/help/how-to-answer).

Draft saved Draft discarded

### Sign up or [log in](https://stats.stackexchange.com/users/login?ssrc=question_page&returnurl=https%3a%2f%2fstats.stackexchange.com%2fquestions%2f249378%2fis-scaling-data-0-1-necessary-when-batch-normalization-is-used%23new-answer)
 Sign up using Google  Sign up using Email and Password Submit

### Post as a guest

Name Email Required, but never shown  Post Your Answer  Discard By clicking “Post Your Answer”, you agree to our [terms of service](https://stackoverflow.com/legal/terms-of-service/public) and acknowledge you have read our [privacy policy](https://stackoverflow.com/legal/privacy-policy).

Start asking to get answers Find the answer to your question by asking.

[Ask question](https://stats.stackexchange.com/questions/ask) Explore related questions
See similar questions with these tags.
*    Featured on Meta
*     [Partnering with Communities to Modernize Policies & Norms](https://meta.stackexchange.com/questions/418826/partnering-with-communities-to-modernize-policies-norms)
### Related

[9](https://stats.stackexchange.com/questions/227114/are-there-any-ways-to-deal-with-the-vanishing-gradient-for-saturating-non-linear "Question score (upvotes - downvotes)")[Are there any ways to deal with the vanishing gradient for saturating non-linearities that doesn't involve Batch Normalization or ReLu units?](https://stats.stackexchange.com/questions/227114/are-there-any-ways-to-deal-with-the-vanishing-gradient-for-saturating-non-linear) [4](https://stats.stackexchange.com/questions/327703/batch-normalization-possible-pros-and-cons-in-one-task "Question score (upvotes - downvotes)")[Batch normalization: possible pros and cons in one task](https://stats.stackexchange.com/questions/327703/batch-normalization-possible-pros-and-cons-in-one-task) [20](https://stats.stackexchange.com/questions/361700/lack-of-batch-normalization-before-last-fully-connected-layer "Question score (upvotes - downvotes)")[Lack of Batch Normalization Before Last Fully Connected Layer](https://stats.stackexchange.com/questions/361700/lack-of-batch-normalization-before-last-fully-connected-layer) [7](https://stats.stackexchange.com/questions/361723/weight-normalization-technique-used-in-image-style-transfer "Question score (upvotes - downvotes)")[Weight normalization technique used in Image Style Transfer](https://stats.stackexchange.com/questions/361723/weight-normalization-technique-used-in-image-style-transfer) [2](https://stats.stackexchange.com/questions/617664/how-does-batch-normalization-enable-larger-learning-rates-according-to-the-orig "Question score (upvotes - downvotes)")[How does batch normalization enable larger learning rates (according to the original paper)?](https://stats.stackexchange.com/questions/617664/how-does-batch-normalization-enable-larger-learning-rates-according-to-the-orig)

#### [Hot Network Questions](https://stackexchange.com/questions?tab=hot)
*    [How/could one figure out any method of variolation for an unknown water born virus without killing too many in the process?](https://worldbuilding.stackexchange.com/questions/273871/how-could-one-figure-out-any-method-of-variolation-for-an-unknown-water-born-vir)
*    [Is there a known obstruction to the prime generating function being modular?](https://math.stackexchange.com/questions/5141946/is-there-a-known-obstruction-to-the-prime-generating-function-being-modular)
*    [Should the bottom of a chromoly steerer tube be closed from water if it's open initially, to prevent rust?](https://bicycles.stackexchange.com/questions/100433/should-the-bottom-of-a-chromoly-steerer-tube-be-closed-from-water-if-its-open-i)
*    [Why is the `..` entry missing in `dir` output for directories directly under a drive root?](https://superuser.com/questions/1938642/why-is-the-entry-missing-in-dir-output-for-directories-directly-under-a-d)
*    [How can I create properly connected Persian/Arabic text using Geometry Nodes?](https://blender.stackexchange.com/questions/347337/how-can-i-create-properly-connected-persian-arabic-text-using-geometry-nodes)
*    [Allocation of mortgage interest deduction between joint homeowners](https://money.stackexchange.com/questions/169747/allocation-of-mortgage-interest-deduction-between-joint-homeowners)
*    [How to disable auto-complete in Libreoffice Writer?](https://superuser.com/questions/1938707/how-to-disable-auto-complete-in-libreoffice-writer)
*    [Beatrice's experiment with mirrors in canto 2 of Dante's "Paradiso"](https://literature.stackexchange.com/questions/32135/beatrices-experiment-with-mirrors-in-canto-2-of-dantes-paradiso)
*    [Advice on Citing a Paper that has been Rejected but is Available on OpenReview?](https://academia.stackexchange.com/questions/227085/advice-on-citing-a-paper-that-has-been-rejected-but-is-available-on-openreview)
*    [Short story about a pair of robots that can reproduce by using electronic components growing like plants. Female repairs male's hand](https://scifi.stackexchange.com/questions/305006/short-story-about-a-pair-of-robots-that-can-reproduce-by-using-electronic-compon)
*    [Where to get Ubuntu Yaru theme assets?](https://askubuntu.com/questions/1567998/where-to-get-ubuntu-yaru-theme-assets)
*    [Water softener drain line discharge height](https://diy.stackexchange.com/questions/331089/water-softener-drain-line-discharge-height)
*    [What is the name of this 3D solid?](https://math.stackexchange.com/questions/5141890/what-is-the-name-of-this-3d-solid)
*    [How to disable Firefox un-maximizing window via double clicking the tab toolbar?](https://superuser.com/questions/1938680/how-to-disable-firefox-un-maximizing-window-via-double-clicking-the-tab-toolbar)
*    [Can I go back and get the Chapter 5 egg after missing it?](https://gaming.stackexchange.com/questions/419078/can-i-go-back-and-get-the-chapter-5-egg-after-missing-it)
*    [How is learning metaphysics useful for spiritual progress?](https://hinduism.stackexchange.com/questions/70137/how-is-learning-metaphysics-useful-for-spiritual-progress)
*    [Why did Littlefinger do this?](https://movies.stackexchange.com/questions/132069/why-did-littlefinger-do-this)
*    [At what age is it optimal to repot plants?](https://gardening.stackexchange.com/questions/70857/at-what-age-is-it-optimal-to-repot-plants)
*    [Spacing TikZ pictures with \hspace and \vspace](https://tex.stackexchange.com/questions/764207/spacing-tikz-pictures-with-hspace-and-vspace)
*    [How to show text data in TCP packets in Wireshark?](https://superuser.com/questions/1938662/how-to-show-text-data-in-tcp-packets-in-wireshark)
*    [In discussing end behavior of polynomials, is there some reason to ask the sign of the constant?](https://matheducators.stackexchange.com/questions/30270/in-discussing-end-behavior-of-polynomials-is-there-some-reason-to-ask-the-sign)
*    [Interaction between drawing something in the background and a page of floats](https://tex.stackexchange.com/questions/764189/interaction-between-drawing-something-in-the-background-and-a-page-of-floats)
*    [Is social commentary a requirement for sci-fi?](https://writing.stackexchange.com/questions/72630/is-social-commentary-a-requirement-for-sci-fi)
*    [What is the reason to argue for and use "synthetic data"?](https://scicomp.stackexchange.com/questions/45474/what-is-the-reason-to-argue-for-and-use-synthetic-data)

[more hot questions](https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used#) [Question feed](https://stats.stackexchange.com/feeds/question/249378 "Feed of this question and its answers")

# Subscribe to RSS

 Question feed To subscribe to this RSS feed, copy and paste this URL into your RSS reader.

[](https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used#)

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
![Image 11: Stack Exchange Inc.](https://cdn.cookielaw.org/logos/static/ot_company_logo.png)

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

![Image 13: .](https://ams-pageview-public.s3.amazonaws.com/1x1-pixel.png?id=b1ffe3826ebc)

<p class="kb-image-caption">图例</p>
