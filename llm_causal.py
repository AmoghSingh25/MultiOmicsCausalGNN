import numpy as np
import os
from rag_utils import LLM_RAG
from tqdm import tqdm


def llm_prots_cd(prots, rag, entity_type="Protein"):
    prompt = f"""
    From the given {entity_type}s, link them together in a way that you believe is the accurate causal relation between them and indicates a interaction according to you. Each {entity_type} is on a single line with a short description. Return a list of pairs, and each pair should be seperated by ';'. Each pair must contain of exactly two {entity_type}s only from the input list, seperated by a ','.
        Do not provide any additional information apart from the answer in <output> tags.
        Assume any additional context and description of parameters. You asbolutely have to surround your answer in <output> tags like so.
        <output>
        ANSWER
        </output>
        The proteins are 
        {"\n".join(prots)}
    """
    response = rag.run_query(prompt)["answer"]
    return response


def check_llm(edges, rag, entity_type="Protein", use_rag=True):
    prompt = f"""
        Each line shows a pair of {entity_type}. Using the information given and your knowledge, answer if the {entity_type}s could have an interaction. The {entity_type}s are seperated using a space.
        Do not provide any additional information apart from the answer in <output> tags.
        Assume additional context and the description of the parameters and utilize your existing knowledge. You asbolutely have to surround your answer in <output> tags like so. Answer using one of the three options, 1-The {entity_type}s could have an interaction, 2-There can be no interaction or 3-Not sure. You have to answer with one of the three numbers which represent the options.
        The output must be a line of integers seperated by ';'.
        <output>
        ANSWER
        </output>
        The pairs are as given below - 
        {"\n".join([x[0] + "," + x[1] for x in edges])}
    """
    if use_rag:
        output = rag.run_rag_query(prompt, [y for x in edges for y in x])["answer"]
    else:
        output = rag.run_query(query=prompt)["answer"]
    return output


def orient_llm(edges, rag, entity_type="Protein", use_rag=True):
    prompt = f"""
        Each line shows a pair of {entity_type}. Using the information given and your knowledge, answer if the {entity_type}s direction of causation is correct. The first parameter is the cause and the second parameter is the effect. The {entity_type}s are seperated using a space.
        Do not provide any additional information apart from the answer in <output> tags.
        Assume additional context and the description of the parameters and utilize your existing knowledge. You asbolutely have to surround your answer in <output> tags like so. Answer using one of the three options, 1-The causal relation is correct, 2-The causal relation is reversed, 3-Not Sure. You have to answer with one of the three numbers which represent the options.
        The output must be a line of integers seperated by ';'.
        <output>
        ANSWER
        </output>
        The pairs are as given below - 
        {"\n".join([x[0] + "," + x[1] for x in edges])}
    """
    if use_rag:
        output = rag.run_rag_query(prompt, [y for x in edges for y in x])["answer"]
    else:
        output = rag.run_query(query=prompt)["answer"]
    return output


def basic_output_process(resp):
    start_token = "<output>"
    end_token = "</output>"
    if start_token not in resp or end_token not in resp:
        return None
    resp = resp[resp.index(start_token) + len(start_token) : resp.index(end_token)]
    resp = resp.replace("\n", " ").split(";")
    return resp


def process_check_output(resp, edges):
    options = basic_output_process(resp)
    if options is None:
        return options
    valid_edges = []
    for i in range(len(options)):
        options[i] = int(options[i].strip())
        if options[i] < 1 or options[i] > 3:
            continue
        if options[i] == 1 or options[i] == 3:
            valid_edges.append(edges[i])
    print(
        "\tIncorrect options in LLM output for check = ", len(edges) - len(valid_edges)
    )
    return valid_edges


def process_cd_output(resp, labels):
    pairs = basic_output_process(resp)
    adj_matrix = np.zeros((len(labels), len(labels)))
    incorr_edges = 0
    for i in range(len(pairs)):
        pairs[i] = pairs[i].split(",")
        if len(pairs[i]) == 2:
            cause_node = pairs[i][0].strip()
            effect_node = pairs[i][1].strip()
            if cause_node in labels and effect_node in labels:
                adj_matrix[labels.index(effect_node)][labels.index(cause_node)] = 1
            else:
                incorr_edges += 1
    print("\tIncorrect edges in LLM output for CD = ", incorr_edges)
    return adj_matrix


def process_orient_output(resp, edges):
    print("Orient Output - - - ")
    print(edges)
    options = basic_output_process(resp)
    print(options)
    valid_edges = []
    for i in range(len(options)):
        options[i] = int(options[i].strip())
        if options[i] < 1 and options[i] > 3:
            continue
        if options[i] == 1 or options[i] == 3:
            valid_edges.append(edges[i])
        else:
            valid_edges.append((edges[i][1], edges[i][0]))
    print(
        "\tIncorrect options in LLM output for orientation = ",
        len(edges) - len(valid_edges),
    )
    return valid_edges


def run_llm(
    undirected_edges,
    directed_edges,
    input_data,
    objective,
    use_rag,
    rag_name,
    llm_model_id,
    temperature,
    entity_type,
    output_dir,
):
    if objective is None:
        undirected_edges.extend(directed_edges)
        print("\tNo objective for LLM")
        return undirected_edges
    if objective == "orient" and len(undirected_edges) == 0:
        print("\tNo undirected edges to orient using LLM...")
        undirected_edges.extend(directed_edges)
        return undirected_edges
    rag_path = os.path.join(output_dir, rag_name)
    rag = LLM_RAG(model_id=llm_model_id, temperature=temperature)
    labels = [f"{col}" for i, col in enumerate(input_data.columns)]
    if os.path.exists(rag_path):
        print("\tLoading from RAG vector store")
        rag.load_memory_vector(rag_path)
    else:
        if use_rag:
            ## Perform wikipedia search for docs
            print("\tScraping for RAG docs")
            for i in tqdm(labels):
                rag.get_wiki_page_and_store(i)
            print("\tSaving RAG vectors..")
            rag.save_memory_vector(rag_path)
        else:
            print("Not using RAG")

    objective = objective.lower()
    if objective == "check":
        print("\tChecking edges using LLM")
        all_edges = undirected_edges
        all_edges.extend(directed_edges)
        resp = check_llm(
            edges=all_edges, rag=rag, entity_type=entity_type, use_rag=use_rag
        )
        valid_edges = process_check_output(resp=resp, edges=all_edges)
        return valid_edges
    elif objective == "orient_check":
        print("\tPerforming orientation and checking using LLM")
        valid_edges = []
        if len(undirected_edges) > 0:
            resp = orient_llm(
                edges=undirected_edges,
                rag=rag,
                entity_type=entity_type,
                use_rag=use_rag,
            )
            valid_edges = process_orient_output(resp=resp, edges=undirected_edges)
        valid_edges.extend(directed_edges)
        resp = check_llm(
            edges=valid_edges, rag=rag, entity_type=entity_type, use_rag=use_rag
        )
        valid_edges = process_check_output(resp=resp, edges=valid_edges)
        return valid_edges
    elif objective == "orient":
        print("\tPerforming Orientation")
        resp = orient_llm(
            edges=undirected_edges, rag=rag, entity_type=entity_type, use_rag=use_rag
        )
        valid_edges = process_orient_output(resp=resp, edges=undirected_edges)
        return valid_edges
    elif objective == "cd":
        print("\tPerforming CD using LLM")
        resp = llm_prots_cd(prots=labels, rag=rag, entity_type=entity_type)
        return process_cd_output(resp=resp, labels=labels)
    else:
        print("\tInvalid objective")
        raise Exception("Invalid LLM Objective")
