import streamlit as st
import config
import utils

title = "Modélisation"
sidebar_name = "Modélisation"

def run():
    st.image(config.get_asset_path("collaboration.svg"))

    st.title(title)
    
    tab1, tab2, tab3 = st.tabs(["Description des données", "Échantillon exemple", "Pipeline de machine learning"])
    with tab1:
        st.markdown("### Description des données textuelles")
        st.markdown(
            """
            | Nom de la colonne | Description                                                                                                       | Disponibilité | Type informatique | Taux de NA |
            |-------------------|-------------------------------------------------------------------------------------------------------------------|---------------|-------------------|------------|
            | designation       | L'appelation du produit | Toujours      | chaîne            | 0,00 %     |
            | description       | Description plus détaillée du produit   | Optionnelle   | chaîne            | 35.09%     |
            | productid         | L'index du produit                                                                                                | Toujours      | int64             | 0,00 %     |
            | imageid           | L'index de l'image                                                                                                | Toujours      | int64             | 0,00 %     |
            """
        )

        st.markdown("### Description des données images")
        st.markdown(
            """
            - Dimension 500x500 pixels
            - Couleurs en RGB
            """
        )

        st.markdown("### Description de la variable cible")
        st.markdown(
            """
            | Nom de la colonne | Description          | Disponibilité | Type informatique | Taux de NA | Distribution des valeurs    |
            |-------------------|----------------------|---------------|-------------------|------------|-----------------------------|
            | prdtypecode       | Catégorie du produit | Toujours      | numérique         | 0,00 %     | Variable catégorielle cible |
            """
        )
        st.markdown(
            """
            Il existe 27 classes de produits distinctes dans le jeu de données d'apprentissage.
            Pour faciliter la compréhension des données, nous avons décidé de les nommer nous même.

            | prdtypecode | label                                  |
            |-------|----------------------------------------------|
            | 10    | Livres et ouvrages culturels                 |
            | 40    | Jeux vidéo et accessoires                    |
            | 50    | Accessoires gaming                           |
            | 60    | Consoles rétro                               |
            | 1140  | Figurines Pop & licences geek                |
            | 1160  | Cartes à collectionner                       |
            | 1180  | Jeux de figurines & wargames                 |
            | 1280  | Jouets enfants & bébés                       |
            | 1281  | Jeux et loisirs enfants                      |
            | 1300  | Drones et modèles réduits                    |
            | 1301  | Chaussettes & accessoires enfants            |
            | 1302  | Jouets divers & loisirs créatifs             |
            | 1320  | Puériculture & équipement bébé               |
            | 1560  | Mobilier & articles de maison                |
            | 1920  | Linge de maison & décoration textile         |
            | 1940  | Alimentation & boissons                      |
            | 2060  | Décoration & accessoires saisonniers         |
            | 2220  | Accessoires pour animaux                     |
            | 2280  | Magazines & journaux anciens                 |
            | 2403  | Livres, mangas & partitions                  |
            | 2462  | Lots jeux vidéo et consoles                  |
            | 2522  | Fournitures de papeterie                     |
            | 2582  | Mobilier et accessoires de jardin            |
            | 2583  | Accessoires pour piscines et spas            |
            | 2585  | Outils et équipements de jardinage           |
            | 2705  | Essais & livres d’histoire                   |
            | 2905  | Jeux PC à télécharger & éditions spéciales   |
            """
        )
    with tab2:
        X_train, y_train = utils.load_data()
        # only display 10 rows
        X_train = X_train.head(10)
        for index, row in X_train.iterrows():
            st.markdown("---")
            cols = st.columns(2)
            with cols[0]:
                st.image(config.get_data_path(f"images/image_{row['imageid']}_product_{row['productid']}.jpg"))
            with cols[1]:
                st.markdown(
                    f"""
                    **Catégorie de produit :** {y_train.loc[index, 'prdtypecode_label']}

                    **Product ID :** {row['productid']}

                    **Designation :** {row['designation']}

                    **Description :**
                    """,
                )
                st.html(row['description'])

    with tab3:
        st.markdown("## Pipeline de machine learning")
        st.image(config.get_asset_path("modelisation.drawio.svg"))

        with st.expander("Schéma technique détaillé"):

            svg = open(config.get_asset_path("mermaid-flow.svg")).read()
            # st.image(svg, width=1000)
            utils.render_svg(svg)