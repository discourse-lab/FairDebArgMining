# FairDebArgMining
This is the repository for the argumentation mining experiments on the dataset of secondary-level argumentative student essays collected through the "Fair Debating and Written Argumentation" (FDE) project. Details on the project are described in:

- Winnie-Karen Giera, Manfred Stede, Lucas Deutzmann, and Eric Graßnick. 2025a. Exploring the Power of Persuasion in Written Argumentation: A Mixed-Methods Pilot Study (QASA). *Journal of Applied Language Learning*, 2(2):1–10.
- Winnie-Karen Giera, Lucas Deutzmann, and Subhan Sheikh Muhammad. 2025b. Merging oral and written argumentation: Supporting student writing through debate and srsd in inclusive classrooms. *Education Sciences*, 15(11):1471.

# Extension of FDE with argumentation annotation (FDE-Arg)

We introduce an annotation scheme for analysing secondary-level argumentative student essays of varying writing qualities, on the basis of the FDE dataset. In this context, we introduce a fine-grained set of argument component types (AC types) that serves as a first step to a formal description of the argumentation structure within an essay. Furthermore, We annotate the argumentative relations that hold between the argumentative components. Our annotation process took place in multiple phases, with minor revisions of the scheme. This yields different versions of annotations as described below.

Our overall scheme, including theoretical and practical motivations for our annotation decisions, are described in detail in:

- Xiaoyu Bai, Kemal Afzal, Dietmar Benndorf, Lucas Deutzmann, Winnie-Karen Giera, Eric Graßnick, and Manfred Stede. 2026. From newspapers to classrooms: Adapting an annotation scheme and automatic classifiers to mixed-quality argumentative school essays. *Argument & Computation*. In press.

## Version 1

This is the first and original version of the annotation scheme and the basis of Bai et al. (2026).

This version only includes argument component type (AC type) annotations and does not include annotations of argument relations.

The annotation guideline for this version (at present only available in German) can be found here: `data/v1/AnnoRiLi_Schueleraufsaetze-V1.pdf`<br>
An inter-annotator agreement study of our AC type annotations based on 30 essays annotated according to this version can be found here: `data/v1/iaa_study/ac_types/`<br>
A set of 50 essays annotated according to this version can be found here:`data/v1/gold_data/ac_types`<br>

## Version 2

We have made slight changes to our annotation of AC types and added argument relation annotations. Data using this annotation version is the basis of the following publication:

- Xiaoyu Bai and Manfred Stede. 2026. Fine-Grained Content Zone Prediction in German Argumentative Essays Using LLMs.
  
This paper will be presented at the *21st Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2026)* in San Diego, United States, and is due to be published in its proceedings.

Please note that for reasons of compatibility with earlier work, different terminology is used in Bai & Stede (2026), where AC types are referred to as "argumentative content zones".

The annotation guideline for this version (at present only available in German) can be found here: `data/v2/AnnoRiLi_Schueleraufsaetze-V2.pdf` <br>
At present, only the 100-sample dataset used by Bai & Stede (2026) is publicly available and can be found here: `data/v2/gold_data/ac_types`. This does not yet include annotations for argument relations.<br>

We expect to release the full dataset (consisting of approximately 1010 essays) along with all of our annotations (including argument relation annotations) once all annotation work has been completed.

## Automatic prediction of argument component types / argumentative content zones using LLMs

We conducted experiments on using LLMs to automatically predict the segment-level component types / content zones. The resources used are here: `experiments/ac_type_prediction`<br>
