# this script concern the analysis of datasets containing semantic informations about domanins, ranges, and classes
import pandas as pd
# controllare se le entità abbiano almeno una classe
dataset_name = 'DBpediaYAGO'
print(dataset_name)
entities_to_id = pd.read_csv(f"../datasets/{dataset_name}/entity2id.txt", sep='\t', header=None, names=["entity", "id"])
entities_to_class = pd.read_csv(f"../datasets/{dataset_name}/entity2class.txt", sep='\t', header=None, names=["entity_id", "class_id"])

ent_ids = set(entities_to_id["id"].values)
print(f"Num entites: {len(ent_ids)}")
ent_id_w_class = entities_to_class["entity_id"].values
ent_id_w_class = set(ent_id_w_class) # ids of entities with at least one class
print(f"Num entities with (at least) one class: {len(ent_id_w_class)}")
without_class = ent_ids.difference(ent_id_w_class)
print(f"Num entities without any class: {len(without_class)}")
perc_ent_w_class = (len(ent_id_w_class)/len(ent_ids)) * 100
print(f"Percentage of entites that has classes: {round(perc_ent_w_class,2)} %")

print()

# controllare se le  relazioni abbiano almeno un dominio
relations_to_id = pd.read_csv(f"../datasets/{dataset_name}/relation2id.txt", sep='\t', header=None, names=["relation", "id"])
"""Attenzione, sono separati da spazio e non da tab"""
relations_to_domain = pd.read_csv(f"../datasets/{dataset_name}/rs_domain2id.txt", sep=' ', header=None, names=["rel_id", "domain_id"])

rel_ids = set(relations_to_id["id"].values)
print(f"Num relations: {len(rel_ids)}")
rel_id_w_domain = relations_to_domain["rel_id"].values
rel_id_w_domain = set(rel_id_w_domain) # ids of entities with at least one class
print(f"Num relations with (at least) one domain: {len(rel_id_w_domain)}")
without_domain = rel_ids.difference(rel_id_w_domain)
print(f"Num relations without any domain: {len(without_domain)}")
perc_rel_w_dom = (len(rel_id_w_domain) / len(rel_ids)) * 100
print(f"Percentage of relations that has domains: {round(perc_rel_w_dom, 2)} %")
print()
# controllare se le  relazioni abbiano almeno un range
relations_to_range = pd.read_csv(f"../datasets/{dataset_name}/rs_range2id.txt", sep=' ', header=None, names=["rel_id", "range_id"])
print(f"Num relations: {len(rel_ids)}")
rel_id_w_range = relations_to_range["rel_id"].values
rel_id_w_range = set(rel_id_w_range) # ids of entities with at least one class
print(f"Num relations with (at least) one range: {len(rel_id_w_range)}")
without_range = rel_ids.difference(rel_id_w_range)
print(f"Num relations without any range: {len(without_range)}")
perc_rel_w_dom = (len(rel_id_w_range) / len(rel_ids)) * 100
print(f"Percentage of relations that has ranges: {round(perc_rel_w_dom, 2)} %")
# salvataggio di esempio, da adattare se serve
# entities2id.to_csv(save_folder + 'entity2id.txt', sep='\t', header=None, index=None)
rel_range_dom = rel_id_w_range.intersection(rel_id_w_domain)
rel_range_dom_perc = round((len(rel_range_dom)/len(rel_ids)) *100, 2)
print(f"Relations with at least one domain AND one range: {len(rel_range_dom)}\n Perc: {rel_range_dom_perc}%")